//
//  LaneDetectionService.swift
//  WheelBuddy
//
//  Experimental geometric lane tracker from MultiCam rear frames.
//  Not steering control — visual tracking + drift warning only.
// Frame-level perception is delegated to `LanePerceptionBackend` (default: CoreMLLaneBackend with geometric fallback).
//

import Combine
import CoreVideo
import Foundation

/// Consumes existing MultiCam rear `CVPixelBuffer`s (~5 Hz) and publishes lane assist state.
@MainActor
final class LaneDetectionService: ObservableObject {
    @Published private(set) var result: LaneTrackingResult = .empty
    @Published private(set) var isRunning = false

    #if DEBUG
    @Published private(set) var debugSnapshot = LaneDebugSnapshot()
    #endif

    // MARK: - Tunable constants (temporal / state — do not move into the backend)

    /// Analysis rate target.
    nonisolated static let minInterval: TimeInterval = 0.20 // ~5 Hz

    nonisolated static let vehicleCenterX: CGFloat = 0.50

    nonisolated static let emaAlpha: CGFloat = 0.30

    /// Require stronger confidence to initially acquire lanes.
    nonisolated static let confidenceAcquireThreshold: CGFloat = 0.45

    /// Once acquired, tolerate short confidence dips before declaring lanes lost.
    nonisolated static let confidenceKeepThreshold: CGFloat = 0.28

    /// How long an already-acquired lane may stay weak before becoming unavailable.
    nonisolated static let confidenceLossGraceDuration: TimeInterval = 0.60

    nonisolated static let driftEnterOffset: CGFloat = 0.55
    nonisolated static let driftExitOffset: CGFloat = 0.40
    nonisolated static let driftEnterDuration: TimeInterval = 0.70
    nonisolated static let driftExitDuration: TimeInterval = 0.40

    /// Lane-departure WARNING only at/above this GPS speed (MPH).
    nonisolated static let warningMinSpeedMPH: Double = 20.0

    // MARK: - Runtime

    private let backend: any LanePerceptionBackend

    private let processingQueue = DispatchQueue(label: "com.wheelbuddy.lane.vision", qos: .userInitiated)
    private let processingLock = NSLock()
    private nonisolated(unsafe) var isProcessingFrame = false
    private nonisolated(unsafe) var lastAnalysisUptime: TimeInterval = 0
    private nonisolated(unsafe) var timingLogCounter = 0
    private nonisolated(unsafe) var didLogStartup = false

    /// Latest GPS speed for warning gate (updated from DriveView / SpeedMonitor).
    private var latestSpeedMPH: Double?
    private var latestSpeedReliable = false

    // Smoothed telemetry (MainActor)
    private var smoothLeftX: CGFloat?
    private var smoothRightX: CGFloat?
    private var smoothOffset: CGFloat = 0
    private var smoothConfidence: CGFloat = 0
    private var publishedState: LaneAssistState = .unavailable

    private var candidateDriftSide: LaneAssistState? // .driftingLeft / .driftingRight only
    private var candidateDriftSince: TimeInterval?
    private var centeredSince: TimeInterval?
    private var lowConfidenceSince: TimeInterval?

    private var previousStateForLog: LaneAssistState = .unavailable

    init(
        backend: any LanePerceptionBackend = CoreMLLaneBackend()
    ) {
        self.backend = backend
    }

    func beginExternalFrameProcessing() {
        isRunning = true
        resetTracking()
        #if DEBUG
        if !didLogStartup {
            didLogStartup = true
            print("[Lane] rear frame processing active 5Hz")
        }
        #else
        if !didLogStartup {
            didLogStartup = true
            print("[Lane] rear frame processing active 5Hz")
        }
        #endif
    }

    func endExternalFrameProcessing() {
        isRunning = false
        resetTracking()
        result = .empty
    }

    /// Call when GPS speed updates (does not re-run Vision).
    func updateSpeed(mph: Double?, reliable: Bool) {
        latestSpeedMPH = mph
        latestSpeedReliable = reliable
        // Re-evaluate warning eligibility if currently drifting / tracking.
        if isRunning {
            republishWithCurrentSpeed()
        }
    }

    /// MultiCam rear callback — must return quickly; analysis is async on `processingQueue`.
    nonisolated func processExternalFrame(_ pixelBuffer: CVPixelBuffer) {
        let now = ProcessInfo.processInfo.systemUptime
        if now - lastAnalysisUptime < Self.minInterval {
            return
        }

        processingLock.lock()
        if isProcessingFrame {
            processingLock.unlock()
            return
        }
        isProcessingFrame = true
        lastAnalysisUptime = now
        processingLock.unlock()

        let retainedBuffer = pixelBuffer
        processingQueue.async { [weak self] in
            defer {
                self?.processingLock.lock()
                self?.isProcessingFrame = false
                self?.processingLock.unlock()
            }
            guard let self else { return }
            let started = ProcessInfo.processInfo.systemUptime
            let perception = self.backend.analyze(pixelBuffer: retainedBuffer)
            let elapsedMs = (ProcessInfo.processInfo.systemUptime - started) * 1000
            self.timingLogCounter += 1
            if self.timingLogCounter % 15 == 1 {
                print(String(format: "[Lane] analysis %.1f ms", elapsedMs))
            }
            Task { @MainActor in
                self.ingest(perception: perception, uptime: ProcessInfo.processInfo.systemUptime)
            }
        }
    }

    // MARK: - Temporal / publish (MainActor)

    private func resetTracking() {
        smoothLeftX = nil
        smoothRightX = nil
        lowConfidenceSince = nil
        smoothOffset = 0
        smoothConfidence = 0
        publishedState = .unavailable
        candidateDriftSide = nil
        candidateDriftSince = nil
        centeredSince = nil
        previousStateForLog = .unavailable
        #if DEBUG
        debugSnapshot = LaneDebugSnapshot()
        #endif
    }

    private func ingest(perception: LanePerceptionResult, uptime: TimeInterval) {
        guard isRunning else { return }

        // EMA confidence
        smoothConfidence = Self.emaAlpha * perception.confidence + (1 - Self.emaAlpha) * smoothConfidence

        if let lx = perception.leftXAtEvaluationY {
            smoothLeftX = smoothLeftX.map { Self.emaAlpha * lx + (1 - Self.emaAlpha) * $0 } ?? lx
        }
        if let rx = perception.rightXAtEvaluationY {
            smoothRightX = smoothRightX.map { Self.emaAlpha * rx + (1 - Self.emaAlpha) * $0 } ?? rx
        }

        // Decay when a side is missing this frame.
        if perception.leftXAtEvaluationY == nil { smoothConfidence *= 0.92 }
        if perception.rightXAtEvaluationY == nil { smoothConfidence *= 0.92 }

        var offset: CGFloat = 0
        var laneCenter: CGFloat?
        if let l = smoothLeftX, let r = smoothRightX, r > l {
            let width = r - l
            let center = (l + r) * 0.5
            laneCenter = center
            let half = max(width * 0.5, 1e-3)
            // centerError > 0 when lane center is left of vehicle → vehicle is right of lane center.
            let centerError = Self.vehicleCenterX - center
            offset = centerError / half
            offset = max(-1.25, min(1.25, offset))
        } else {
            smoothConfidence *= 0.85
        }

        smoothOffset = Self.emaAlpha * offset + (1 - Self.emaAlpha) * smoothOffset
        smoothOffset = max(-1.2, min(1.2, smoothOffset))

        let nextState = resolveState(uptime: uptime)

        // If tracking was acquired but is now genuinely lost,
        // remove stale lane geometry before the next acquisition.
        if nextState == .unavailable,
        publishedState != .unavailable {

            smoothLeftX = nil
            smoothRightX = nil
            smoothOffset = 0
            laneCenter = nil
        }

        #if DEBUG
        debugSnapshot = LaneDebugSnapshot(
            leftX: smoothLeftX,
            rightX: smoothRightX,
            laneCenterX: laneCenter,
            confidence: smoothConfidence,
            lateralOffset: smoothOffset,
            leftLanePoints: perception.leftLanePoints,
            rightLanePoints: perception.rightLanePoints,
            frameConfidence: perception.confidence
        )
        #endif

        logTransitions(next: nextState)

        publishedState = nextState
        result = LaneTrackingResult(
            state: nextState,
            lateralOffset: smoothOffset,
            confidence: smoothConfidence,
            leftLaneBottomX: smoothLeftX,
            rightLaneBottomX: smoothRightX,
            laneCenterX: laneCenter
        )
    }

    private func resolveState(uptime: TimeInterval) -> LaneAssistState {
        let confidenceThreshold: CGFloat =
            publishedState == .unavailable
                ? Self.confidenceAcquireThreshold
                : Self.confidenceKeepThreshold

        let perceptionUnavailable =
            smoothConfidence < confidenceThreshold
            || smoothLeftX == nil
            || smoothRightX == nil

        if perceptionUnavailable {

            // If we have never acquired lanes yet, stay unavailable.
            if publishedState == .unavailable {
                lowConfidenceSince = nil
                candidateDriftSide = nil
                candidateDriftSince = nil
                centeredSince = nil
                return .unavailable
            }

            // We were tracking before. Start a short grace timer.
            if lowConfidenceSince == nil {
                lowConfidenceSince = uptime
            }

            if let started = lowConfidenceSince,
                uptime - started < Self.confidenceLossGraceDuration {

                // Keep visual tracking alive during a short dropout,
                // but do not preserve a stale drift warning.
                return .tracking
            }

            // Weak for too long -> actually lose tracking.
            lowConfidenceSince = nil
            candidateDriftSide = nil
            candidateDriftSince = nil
            centeredSince = nil
            return .unavailable
        }

        // Good frame again: cancel any pending loss timer.
        lowConfidenceSince = nil

        let absOff = abs(smoothOffset)
        let warningSpeedOK =
            latestSpeedReliable
            && (latestSpeedMPH ?? 0) >= Self.warningMinSpeedMPH

        // Hysteresis clear toward tracking.
        if publishedState == .driftingLeft || publishedState == .driftingRight {
            if absOff < Self.driftExitOffset {
                if centeredSince == nil { centeredSince = uptime }
                if let started = centeredSince,
                   uptime - started >= Self.driftExitDuration {
                    candidateDriftSide = nil
                    candidateDriftSince = nil
                    centeredSince = nil
                    #if DEBUG
                    print("[Lane] centered")
                    #endif
                    return .tracking
                }
                return publishedState
            } else {
                centeredSince = nil
                return publishedState
            }
        }

        // Enter drift only with speed gate + persistence.
        if warningSpeedOK, absOff >= Self.driftEnterOffset {
            let side: LaneAssistState = smoothOffset <= -Self.driftEnterOffset
                ? .driftingLeft
                : .driftingRight

            if candidateDriftSide != side {
                candidateDriftSide = side
                candidateDriftSince = uptime
                #if DEBUG
                print(side == .driftingLeft ? "[Lane] candidate driftLeft" : "[Lane] candidate driftRight")
                #endif
            } else if let started = candidateDriftSince,
                      uptime - started >= Self.driftEnterDuration {
                return side
            }
            return .tracking
        }

        candidateDriftSide = nil
        candidateDriftSince = nil
        centeredSince = nil
        return .tracking
    }

    private func republishWithCurrentSpeed() {
        let uptime = ProcessInfo.processInfo.systemUptime
        // If speed drops below gate while drifting, demote to tracking (keep visual offset).
        if (publishedState == .driftingLeft || publishedState == .driftingRight) {
            let warningSpeedOK =
                latestSpeedReliable
                && (latestSpeedMPH ?? 0) >= Self.warningMinSpeedMPH
            if !warningSpeedOK {
                publishedState = .tracking
                candidateDriftSide = nil
                candidateDriftSince = nil
                centeredSince = nil
                result = LaneTrackingResult(
                    state: .tracking,
                    lateralOffset: result.lateralOffset,
                    confidence: result.confidence,
                    leftLaneBottomX: result.leftLaneBottomX,
                    rightLaneBottomX: result.rightLaneBottomX,
                    laneCenterX: result.laneCenterX
                )
                #if DEBUG
                print("[Lane] drift cleared (speed gate)")
                #endif
                return
            }
        }
        let next = resolveState(uptime: uptime)
        if next != publishedState {
            logTransitions(next: next)
            publishedState = next
            result = LaneTrackingResult(
                state: next,
                lateralOffset: result.lateralOffset,
                confidence: result.confidence,
                leftLaneBottomX: result.leftLaneBottomX,
                rightLaneBottomX: result.rightLaneBottomX,
                laneCenterX: result.laneCenterX
            )
        }
    }

    private func logTransitions(next: LaneAssistState) {
        if previousStateForLog == .unavailable, next == .tracking {
            print(String(format: "[Lane] acquired confidence=%.2f", Double(smoothConfidence)))
        }
        if previousStateForLog != .unavailable, next == .unavailable {
            print(String(format: "[Lane] lost confidence=%.2f", Double(smoothConfidence)))
        }
        if next == .driftingLeft, previousStateForLog != .driftingLeft {
            print(String(format: "[Lane] DRIFT_LEFT offset=%.2f", Double(smoothOffset)))
        }
        if next == .driftingRight, previousStateForLog != .driftingRight {
            print(String(format: "[Lane] DRIFT_RIGHT offset=%.2f", Double(smoothOffset)))
        }
        #if DEBUG
        // Occasional offset sample while tracking.
        if next == .tracking, timingSampleShouldLog() {
            print(String(format: "[Lane] offset=%.2f state=tracking", Double(smoothOffset)))
        }
        #endif
        previousStateForLog = next
    }

    private var debugOffsetLogCounter = 0
    private func timingSampleShouldLog() -> Bool {
        debugOffsetLogCounter += 1
        return debugOffsetLogCounter % 10 == 0
    }
}
