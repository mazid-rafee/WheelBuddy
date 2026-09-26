//
//  DrowsinessInferenceCoordinator.swift
//  WheelBuddy
//

import Combine
import Foundation

/// Buffers ~15 FPS feature rows and POSTs a T=20 window once per second.
/// Publishes wake-up when the remote label is ``closed``, holding the banner for 3s.
final class DrowsinessInferenceCoordinator: ObservableObject {
    /// Latest remote prediction.
    @Published private(set) var latestPrediction: DrowsinessPredictResponse?
    /// True while the wake-up warning banner should be shown (closed detected, held 3s).
    @Published private(set) var isWakeUpAlertActive = false

    private let client = DrowsinessAPIClient()
    private let lock = NSLock()

    private var sessionID = UUID().uuidString
    private var sequenceID = 0
    private var buffer: [DrowsinessFeatureSample] = []
    private var isRunning = false
    private var requestInFlight = false
    private var sendTimer: Timer?
    private var wakeHoldTimer: Timer?
    private var activeTask: Task<Void, Never>?
    private var consecutiveCloseCount = 0

    private let windowFrames = DrowsinessFeatureContract.windowFrames
    private let maxBuffer = DrowsinessFeatureContract.windowFrames * 4
    private let sendInterval: TimeInterval = 1.0
    private let wakeUpCloseThreshold = 4
    private let wakeHoldDuration: TimeInterval = 3.0
    private static let closedLabel = "closed"

    func start() {
        lock.lock()
        sessionID = UUID().uuidString
        sequenceID = 0
        buffer.removeAll(keepingCapacity: true)
        requestInFlight = false
        consecutiveCloseCount = 0
        isRunning = true
        lock.unlock()

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.latestPrediction = nil
            self.isWakeUpAlertActive = false
            self.wakeHoldTimer?.invalidate()
            self.wakeHoldTimer = nil
            self.sendTimer?.invalidate()
            let timer = Timer.scheduledTimer(withTimeInterval: self.sendInterval, repeats: true) { [weak self] _ in
                self?.tickSend()
            }
            RunLoop.main.add(timer, forMode: .common)
            self.sendTimer = timer
        }
    }

    func stop() {
        lock.lock()
        isRunning = false
        buffer.removeAll(keepingCapacity: false)
        requestInFlight = false
        consecutiveCloseCount = 0
        lock.unlock()

        activeTask?.cancel()
        activeTask = nil
        DispatchQueue.main.async { [weak self] in
            self?.latestPrediction = nil
            self?.isWakeUpAlertActive = false
            self?.wakeHoldTimer?.invalidate()
            self?.wakeHoldTimer = nil
            self?.sendTimer?.invalidate()
            self?.sendTimer = nil
        }
    }

    /// Called from the camera / Vision path (any queue).
    nonisolated func ingest(_ sample: DrowsinessFeatureSample) {
        guard sample.isFinite else { return }
        lock.lock()
        defer { lock.unlock() }
        guard isRunning else { return }
        buffer.append(sample)
        if buffer.count > maxBuffer {
            buffer.removeFirst(buffer.count - maxBuffer)
        }
    }

    private func tickSend() {
        lock.lock()
        guard isRunning, !requestInFlight, buffer.count >= windowFrames else {
            lock.unlock()
            return
        }
        let window = Array(buffer.suffix(windowFrames))
        sequenceID += 1
        let seq = sequenceID
        let session = sessionID
        requestInFlight = true
        lock.unlock()

        let request = DrowsinessPredictRequest(
            featureSchemaVersion: DrowsinessFeatureContract.schemaVersion,
            sessionID: session,
            sequenceID: seq,
            sentAtUTC: Date(),
            samplingRateHz: DrowsinessFeatureContract.samplingRateHz,
            featureNames: DrowsinessFeatureContract.featureNames,
            samples: window.map {
                DrowsinessAPISample(timestampMs: $0.timestampMs, values: $0.values)
            }
        )

        let started = Date()
        activeTask = Task { [weak self] in
            guard let self else { return }
            do {
                let response = try await client.predict(request)
                let roundTripMs = Date().timeIntervalSince(started) * 1000.0
                await self.handleSuccess(
                    expectedSession: session,
                    expectedSequence: seq,
                    response: response,
                    roundTripMs: roundTripMs
                )
            } catch is CancellationError {
                await self.clearInFlight()
            } catch {
                await self.handleFailure(
                    expectedSession: session,
                    expectedSequence: seq,
                    error: error
                )
            }
        }
    }

    @MainActor
    private func handleSuccess(
        expectedSession: String,
        expectedSequence: Int,
        response: DrowsinessPredictResponse,
        roundTripMs: Double
    ) {
        lock.lock()
        let sessionMatches = sessionID == expectedSession
        let sequenceMatches = sequenceID == expectedSequence
        let stillRunning = isRunning
        requestInFlight = false
        lock.unlock()

        guard stillRunning, sessionMatches else { return }
        // Ignore stale responses if a newer sequence was already issued.
        guard response.sessionID == expectedSession,
              response.sequenceID == expectedSequence,
              sequenceMatches else {
            return
        }

        latestPrediction = response
        updateWakeUpFromLabel(response.label)
        print(
            "[DrowsinessRemote] session=\(response.sessionID) seq=\(response.sequenceID) "
                + "label=\(response.label) confidence=\(String(format: "%.4f", response.confidence)) "
                + "schema=\(response.featureSchemaVersion ?? "nil") "
                + "model=\(response.modelVersion) server_ms=\(String(format: "%.1f", response.inferenceLatencyMs)) "
                + "round_trip_ms=\(String(format: "%.1f", roundTripMs)) "
                + "wakeUp=\(isWakeUpAlertActive)"
        )
    }

    @MainActor
    private func updateWakeUpFromLabel(_ label: String) {
        lock.lock()
        if label == Self.closedLabel {
            consecutiveCloseCount += 1
        } else {
            consecutiveCloseCount = 0
        }
        let shouldPresent = consecutiveCloseCount >= wakeUpCloseThreshold
        lock.unlock()

        if shouldPresent {
            presentWakeUpAlert(holdDuration: wakeHoldDuration)
        }
        // Non-closed labels do not clear an active hold — the timer owns dismissal.
    }

    /// Show the wake banner and (re)start the 3s hold. Rising edges are observed by DriveView.
    @MainActor
    private func presentWakeUpAlert(holdDuration: TimeInterval) {
        isWakeUpAlertActive = true
        wakeHoldTimer?.invalidate()
        let timer = Timer.scheduledTimer(withTimeInterval: holdDuration, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.isWakeUpAlertActive = false
                self.wakeHoldTimer = nil
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        wakeHoldTimer = timer
    }

    @MainActor
    private func handleFailure(
        expectedSession: String,
        expectedSequence: Int,
        error: Error
    ) {
        lock.lock()
        requestInFlight = false
        // Failures are not labeled replies — clear streak only.
        // Do not dismiss an active wake hold on network error.
        consecutiveCloseCount = 0
        lock.unlock()
        print(
            "[DrowsinessRemote][ERROR] session=\(expectedSession) seq=\(expectedSequence) "
                + "error=\(error.localizedDescription)"
        )
    }

    @MainActor
    private func clearInFlight() {
        lock.lock()
        requestInFlight = false
        lock.unlock()
    }
}
