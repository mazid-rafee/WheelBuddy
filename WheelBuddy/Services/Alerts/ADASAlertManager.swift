//
//  ADASAlertManager.swift
//  WheelBuddy
//
//  Centralized ADAS audio alerts.
//  Driven only by meaningful state transitions — not SwiftUI recomputes.
//
//  Audio architecture:
//    start → configure session once → activate once → prepare/start engine once
//    beeps → schedule PCM buffers on a persistent AVAudioPlayerNode (no setActive)
//    stop → stop engine → deactivate session once
//

import AVFoundation
import Combine
import Foundation
import UIKit

/// Extensible alert severity for current and future ADAS modules.
enum ADASAlertPriority: Int, Comparable {
    case none = 0
    case informational = 1
    case caution = 2
    case critical = 3

    static func < (lhs: ADASAlertPriority, rhs: ADASAlertPriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// Owns beep patterns, HIGH haptic, and repeating cadence for driver / road / lane ADAS states.
@MainActor
final class ADASAlertManager: ObservableObject {
    // MARK: - Cadence (seconds)

    private static let noFaceRepeatInterval: TimeInterval = 4.5
    private static let cautionRepeatInterval: TimeInterval = 3.0
    private static let lookingAwayRepeatInterval: TimeInterval = 2.5
    private static let laneDepartureRepeatInterval: TimeInterval = 2.5
    private static let highRepeatInterval: TimeInterval = 1.25

    private enum ActiveKind: Equatable {
        case none
        case driverNoFace
        case driverWakeUp
        case roadCaution
        case driverLookingAway
        case laneDeparture
        case roadHigh
        /// Pedestrian Close! — highest road-risk alert priority.
        case pedestrianClose
    }

    private enum ToneKind {
        case soft
        case urgent
    }

    private var activeKind: ActiveKind = .none
    private var previousDriver: DriverAttentionState?
    private var previousRoad: RoadRiskState?
    private var previousPedestrian: PedestrianRiskState?
    private var previousLane: LaneAssistState?
    private var previousWakeUp = false

    private var repeatTimer: Timer?
    private var pendingBeepWorkItems: [DispatchWorkItem] = []

    /// HIGH + lane-departure haptic (lookingAway / caution / noFace never vibrate).
    private let notificationHaptic = UINotificationFeedbackGenerator()
    private let heavyImpact = UIImpactFeedbackGenerator(style: .heavy)
    private let mediumImpact = UIImpactFeedbackGenerator(style: .medium)

    private var isRunning = false
    /// True only after the persistent AVAudioEngine is prepared and running.
    private var audioReady = false
    /// One-shot wake beep requested before the engine finished starting.
    private var pendingWakeUpBeep = false

    /// Persistent AVFoundation playback (session + engine); not MainActor-bound.
    private let toneEngine = ADASToneEngine()

    func start() {
        guard !isRunning else { return }
        isRunning = true
        audioReady = false
        pendingWakeUpBeep = false
        notificationHaptic.prepare()
        heavyImpact.prepare()
        mediumImpact.prepare()
        previousDriver = nil
        previousRoad = nil
        previousPedestrian = nil
        previousLane = nil
        previousWakeUp = false
        activeKind = .none
        toneEngine.start { [weak self] in
            Task { @MainActor in
                self?.handleAudioEngineReady()
            }
        }
    }

    func stop() {
        isRunning = false
        audioReady = false
        pendingWakeUpBeep = false
        clearActiveAlert()
        toneEngine.stop()
        previousDriver = nil
        previousRoad = nil
        previousPedestrian = nil
        previousLane = nil
        previousWakeUp = false
    }

    /// Play a one-shot soft beep-beep (e.g. wake-up banner rising edge).
    /// Does not change the active repeating alert kind.
    func playWakeUpBeep() {
        guard isRunning else { return }
        #if DEBUG
        print("[ADASAudio] play wakeUp (one-shot)")
        #endif
        if audioReady {
            playBeepBeep(kind: .soft, gap: 0.14)
        } else {
            pendingWakeUpBeep = true
        }
    }

    /// Call when driver / vehicle / pedestrian / lane / wake-up ADAS state may have changed.
    func update(
        driverAttention: DriverAttentionState,
        roadRisk: RoadRiskState,
        pedestrianRisk: PedestrianRiskState = .clear,
        laneAssist: LaneAssistState,
        wakeUpAlert: Bool = false
    ) {
        guard isRunning else { return }

        let next = resolveActiveKind(
            driver: driverAttention,
            road: roadRisk,
            pedestrian: pedestrianRisk,
            lane: laneAssist,
            wakeUp: wakeUpAlert
        )
        previousDriver = driverAttention
        previousRoad = roadRisk
        previousPedestrian = pedestrianRisk
        previousLane = laneAssist
        previousWakeUp = wakeUpAlert

        // Before the engine is ready: accept state, keep only the current
        // highest-priority resolved alert — never play or queue history.
        if !audioReady {
            activeKind = next
            return
        }

        guard next != activeKind else { return }
        transition(to: next)
    }

    /// Invoked once when the tone engine reports running (off-main setup complete).
    private func handleAudioEngineReady() {
        guard isRunning, !audioReady else { return }
        audioReady = true

        if pendingWakeUpBeep {
            pendingWakeUpBeep = false
            playBeepBeep(kind: .soft, gap: 0.14)
        }

        guard let driver = previousDriver,
              let road = previousRoad,
              let pedestrian = previousPedestrian,
              let lane = previousLane else {
            activeKind = .none
            return
        }
        let next = resolveActiveKind(
            driver: driver,
            road: road,
            pedestrian: pedestrian,
            lane: lane,
            wakeUp: previousWakeUp
        )
        if next == .none {
            activeKind = .none
            return
        }
        transition(to: next)
    }

    // MARK: - Resolution
    // Priority: pedestrianClose > road HIGH > lane departure > lookingAway > caution > wakeUp / noFace > none

    private func resolveActiveKind(
        driver: DriverAttentionState,
        road: RoadRiskState,
        pedestrian: PedestrianRiskState,
        lane: LaneAssistState,
        wakeUp: Bool
    ) -> ActiveKind {
        if pedestrian == .close {
            return .pedestrianClose
        }
        if road == .high {
            return .roadHigh
        }
        if lane == .driftingLeft || lane == .driftingRight {
            return .laneDeparture
        }
        if driver == .lookingAway {
            return .driverLookingAway
        }
        if road == .caution {
            return .roadCaution
        }
        // Same priority slot as noFace — wake-up replaces that alert when active.
        if wakeUp {
            return .driverWakeUp
        }
        if driver == .noFace {
            return .driverNoFace
        }
        return .none
    }

    // MARK: - Transitions

    private func transition(to next: ActiveKind) {
        cancelPendingBeeps()
        stopRepeatTimer()
        activeKind = next

        switch next {
        case .none:
            break
        case .driverNoFace:
            #if DEBUG
            print("[ADASAudio] play noFace")
            #endif
            playBeepBeep(kind: .soft, gap: 0.14)
            startRepeatTimer(interval: Self.noFaceRepeatInterval)
        case .driverWakeUp:
            #if DEBUG
            print("[ADASAudio] wakeUp active (one-shot beep owned by DriveView)")
            #endif
            // Rising-edge beep is played by DriveView via playWakeUpBeep().
            // No repeating cadence — the UI holds the banner for a fixed duration.
            break
        case .roadCaution:
            #if DEBUG
            print("[ADASAudio] play caution")
            #endif
            playBeepBeep(kind: .soft, gap: 0.12)
            startRepeatTimer(interval: Self.cautionRepeatInterval)
        case .driverLookingAway:
            #if DEBUG
            print("[ADASAudio] play lookingAway")
            #endif
            playLookingAwayBurst()
            startRepeatTimer(interval: Self.lookingAwayRepeatInterval)
        case .laneDeparture:
            #if DEBUG
            print("[ADASAudio] play laneDeparture")
            #endif
            playLaneDepartureBurst()
            startRepeatTimer(interval: Self.laneDepartureRepeatInterval)
        case .roadHigh:
            #if DEBUG
            print("[ADASAudio] play high")
            #endif
            playHighAlertBurst()
            startRepeatTimer(interval: Self.highRepeatInterval)
        case .pedestrianClose:
            #if DEBUG
            print("[ADASAudio] play pedestrianClose")
            #endif
            playHighAlertBurst()
            startRepeatTimer(interval: Self.highRepeatInterval)
        }
    }

    private func clearActiveAlert() {
        cancelPendingBeeps()
        stopRepeatTimer()
        activeKind = .none
        toneEngine.resetScheduledTones()
    }

    private func repeatCurrentAlert() {
        switch activeKind {
        case .none:
            break
        case .driverNoFace:
            #if DEBUG
            print("[ADASAudio] play noFace")
            #endif
            playBeepBeep(kind: .soft, gap: 0.14)
        case .driverWakeUp:
            #if DEBUG
            print("[ADASAudio] play wakeUp")
            #endif
            playBeepBeep(kind: .soft, gap: 0.14)
        case .roadCaution:
            #if DEBUG
            print("[ADASAudio] play caution")
            #endif
            playBeepBeep(kind: .soft, gap: 0.12)
        case .driverLookingAway:
            #if DEBUG
            print("[ADASAudio] play lookingAway")
            #endif
            playLookingAwayBurst()
        case .laneDeparture:
            #if DEBUG
            print("[ADASAudio] play laneDeparture")
            #endif
            playLaneDepartureBurst()
        case .roadHigh:
            #if DEBUG
            print("[ADASAudio] play high")
            #endif
            playHighAlertBurst()
        case .pedestrianClose:
            #if DEBUG
            print("[ADASAudio] play pedestrianClose")
            #endif
            playHighAlertBurst()
        }
    }

    // MARK: - Patterns (timing unchanged for existing alerts)

    private func playBeepBeep(kind: ToneKind, gap: TimeInterval) {
        cancelPendingBeeps()
        toneEngine.scheduleTone(kind == .soft ? .soft : .urgent)
        let second = DispatchWorkItem { [weak self] in
            self?.toneEngine.scheduleTone(kind == .soft ? .soft : .urgent)
        }
        pendingBeepWorkItems = [second]
        DispatchQueue.main.asyncAfter(deadline: .now() + gap, execute: second)
    }

    /// lookingAway: sharper triple urgent beep — audio only, no haptic.
    private func playLookingAwayBurst() {
        cancelPendingBeeps()
        toneEngine.scheduleTone(.urgent)
        let gaps: [TimeInterval] = [0.09, 0.18]
        var items: [DispatchWorkItem] = []
        for gap in gaps {
            let item = DispatchWorkItem { [weak self] in
                self?.toneEngine.scheduleTone(.urgent)
            }
            items.append(item)
            DispatchQueue.main.asyncAfter(deadline: .now() + gap, execute: item)
        }
        pendingBeepWorkItems = items
    }

    /// Lane departure: short urgent beep-beep + medium haptic.
    private func playLaneDepartureBurst() {
        cancelPendingBeeps()
        mediumImpact.impactOccurred(intensity: 0.85)
        mediumImpact.prepare()
        playBeepBeep(kind: .urgent, gap: 0.11)
    }

    /// HIGH: aggressive triple urgent beep + strong haptic.
    private func playHighAlertBurst() {
        cancelPendingBeeps()
        notificationHaptic.notificationOccurred(.error)
        heavyImpact.impactOccurred(intensity: 1.0)
        notificationHaptic.prepare()
        heavyImpact.prepare()

        toneEngine.scheduleTone(.urgent)
        let gaps: [TimeInterval] = [0.13, 0.26]
        var items: [DispatchWorkItem] = []
        for gap in gaps {
            let item = DispatchWorkItem { [weak self] in
                self?.toneEngine.scheduleTone(.urgent)
            }
            items.append(item)
            DispatchQueue.main.asyncAfter(deadline: .now() + gap, execute: item)
        }
        pendingBeepWorkItems = items
    }

    // MARK: - Timer

    private func startRepeatTimer(interval: TimeInterval) {
        stopRepeatTimer()
        let timer = Timer(timeInterval: interval, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.repeatCurrentAlert()
            }
        }
        timer.tolerance = min(0.1, interval * 0.05)
        RunLoop.main.add(timer, forMode: .common)
        repeatTimer = timer
    }

    private func stopRepeatTimer() {
        repeatTimer?.invalidate()
        repeatTimer = nil
    }

    private func cancelPendingBeeps() {
        pendingBeepWorkItems.forEach { $0.cancel() }
        pendingBeepWorkItems.removeAll()
    }
}

// MARK: - Persistent AVFoundation tone engine (off MainActor)

/// One-time AVAudioSession + persistent AVAudioEngine/player for ADAS beeps.
/// All session/engine work runs on `queue` — never the main thread.
private final class ADASToneEngine: @unchecked Sendable {
    enum Tone {
        case soft
        case urgent
    }

    private static let sampleRate: Double = 22_050

    private let queue = DispatchQueue(label: "com.wheelbuddy.adas.audio")

    private var didConfigureCategory = false
    private var isSessionActive = false
    private var isEngineRunning = false

    private var engine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var softBuffer: AVAudioPCMBuffer?
    private var urgentBuffer: AVAudioPCMBuffer?

    /// Begins one-time session activate + engine prepare. Calls `onReady` once
    /// on the audio queue after the engine is running (caller hops to MainActor).
    func start(onReady: @escaping @Sendable () -> Void) {
        queue.async { [weak self] in
            self?.configureActivateAndPrepareEngine(onReady: onReady)
        }
    }

    func stop() {
        queue.async { [weak self] in
            self?.tearDownEngineAndSession()
        }
    }

    func scheduleTone(_ tone: Tone) {
        queue.async { [weak self] in
            guard let self,
                  self.isEngineRunning,
                  let player = self.playerNode else {
                return
            }
            let buffer: AVAudioPCMBuffer?
            switch tone {
            case .soft:
                buffer = self.softBuffer
            case .urgent:
                buffer = self.urgentBuffer
            }
            guard let buffer else { return }
            // No AVAudioSession calls here — only schedule on the warm engine.
            if !player.isPlaying {
                player.play()
            }
            player.scheduleBuffer(buffer, completionHandler: nil)
        }
    }

    func resetScheduledTones() {
        queue.async { [weak self] in
            guard let self else { return }
            self.playerNode?.stop()
            if self.isEngineRunning {
                self.playerNode?.play()
            }
        }
    }

    // MARK: Session + engine

    private func configureActivateAndPrepareEngine(onReady: @escaping @Sendable () -> Void) {
        let session = AVAudioSession.sharedInstance()
        do {
            if !didConfigureCategory {
                #if DEBUG
                print("[ADASAudio] configuring mainThread=\(Thread.isMainThread)")
                #endif
                try session.setCategory(
                    .playback,
                    mode: .default,
                    options: [.mixWithOthers, .duckOthers]
                )
                didConfigureCategory = true
            }

            #if DEBUG
            print("[ADASAudio] activating mainThread=\(Thread.isMainThread)")
            #endif

            if #available(iOS 27.0, *) {
                session.activate(options: []) { [weak self] activated, error in
                    self?.queue.async {
                        self?.handleActivationThenPrepareEngine(
                            activated: activated,
                            error: error,
                            onReady: onReady
                        )
                    }
                }
            } else {
                try session.setActive(true, options: [])
                handleActivationThenPrepareEngine(
                    activated: true,
                    error: nil,
                    onReady: onReady
                )
            }
        } catch {
            #if DEBUG
            print("[ADASAudio] error: \(error.localizedDescription)")
            #endif
        }
    }

    private func handleActivationThenPrepareEngine(
        activated: Bool,
        error: Error?,
        onReady: @escaping @Sendable () -> Void
    ) {
        if let error {
            #if DEBUG
            print("[ADASAudio] error: \(error.localizedDescription)")
            #endif
            isSessionActive = false
            return
        }
        guard activated else {
            #if DEBUG
            print("[ADASAudio] error: activation completed without active session")
            #endif
            isSessionActive = false
            return
        }

        isSessionActive = true
        #if DEBUG
        print("[ADASAudio] active")
        #endif
        prepareAndStartEngine(onReady: onReady)
    }

    private func prepareAndStartEngine(onReady: @escaping @Sendable () -> Void) {
        #if DEBUG
        print("[ADASAudio] engineStart mainThread=\(Thread.isMainThread)")
        #endif

        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Self.sampleRate,
            channels: 1,
            interleaved: false
        )
        guard let format else {
            #if DEBUG
            print("[ADASAudio] error: could not create audio format")
            #endif
            return
        }

        softBuffer = Self.makeToneBuffer(
            frequencyHz: 880,
            durationSeconds: 0.09,
            amplitude: 0.45,
            format: format
        )
        urgentBuffer = Self.makeToneBuffer(
            frequencyHz: 1320,
            durationSeconds: 0.08,
            amplitude: 0.75,
            format: format
        )

        let newEngine = AVAudioEngine()
        let newPlayer = AVAudioPlayerNode()
        newEngine.attach(newPlayer)
        newEngine.connect(newPlayer, to: newEngine.mainMixerNode, format: format)

        do {
            newEngine.prepare()
            #if DEBUG
            print("[ADASAudio] engine prepared")
            #endif
            try newEngine.start()
            newPlayer.play()
            engine = newEngine
            playerNode = newPlayer
            isEngineRunning = true
            #if DEBUG
            print("[ADASAudio] engine running")
            #endif
            onReady()
        } catch {
            #if DEBUG
            print("[ADASAudio] error: \(error.localizedDescription)")
            #endif
            isEngineRunning = false
            engine = nil
            playerNode = nil
        }
    }

    private func tearDownEngineAndSession() {
        #if DEBUG
        print("[ADASAudio] deactivating")
        #endif

        playerNode?.stop()
        engine?.stop()
        if let player = playerNode {
            engine?.detach(player)
        }
        playerNode = nil
        engine = nil
        softBuffer = nil
        urgentBuffer = nil
        isEngineRunning = false

        let session = AVAudioSession.sharedInstance()
        if #available(iOS 27.0, *) {
            session.deactivate(options: [.notifyOthersOnDeactivation]) { [weak self] deactivated, error in
                self?.queue.async {
                    self?.finishDeactivation(deactivated: deactivated, error: error)
                }
            }
        } else {
            do {
                try session.setActive(false, options: [.notifyOthersOnDeactivation])
                finishDeactivation(deactivated: true, error: nil)
            } catch {
                finishDeactivation(deactivated: false, error: error)
            }
        }
    }

    private func finishDeactivation(deactivated: Bool, error: Error?) {
        if let error {
            #if DEBUG
            print("[ADASAudio] error: \(error.localizedDescription)")
            #endif
        }
        if deactivated {
            isSessionActive = false
            #if DEBUG
            print("[ADASAudio] inactive")
            #endif
        }
    }

    private static func makeToneBuffer(
        frequencyHz: Double,
        durationSeconds: Double,
        amplitude: Float,
        format: AVAudioFormat
    ) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(Self.sampleRate * durationSeconds)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return nil
        }
        buffer.frameLength = frameCount
        guard let channels = buffer.floatChannelData else { return nil }
        let samples = channels[0]

        let twoPiF = 2.0 * Double.pi * frequencyHz
        let attackFrames = max(1, Int(Self.sampleRate * 0.005))
        let releaseFrames = max(1, Int(Self.sampleRate * 0.02))
        let total = Int(frameCount)

        for i in 0..<total {
            let t = Double(i) / Self.sampleRate
            var envelope: Float = 1.0
            if i < attackFrames {
                envelope = Float(i) / Float(attackFrames)
            } else if i > total - releaseFrames {
                envelope = Float(total - i) / Float(releaseFrames)
            }
            samples[i] = Float(sin(twoPiF * t)) * amplitude * envelope
        }
        return buffer
    }
}
