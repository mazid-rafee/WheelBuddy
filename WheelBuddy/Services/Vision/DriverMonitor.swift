//
//  DriverMonitor.swift
//  WheelBuddy
//

import Combine
import Foundation
import Vision

/// Processes front-camera frames and publishes driver attention state.
/// Also extracts drowsiness model features (Vision landmarks + hand pose) for remote inference.
@MainActor
final class DriverMonitor: ObservableObject {
    /// Debounced attention state: `.lookingAway` only after `lookingAwayAlertDuration` of sustained yaw.
    @Published private(set) var attentionState: DriverAttentionState = .noFace
    /// Set when the legacy `CameraManager` path fails to start; never set in external-frame mode.
    @Published private(set) var cameraError: CameraError?
    /// Attention updates from Vision results arriving while false are discarded.
    @Published private(set) var isRunning = false

    /// Optional remote drowsiness path; receives one feature sample per analyzed frame.
    /// Its wake-up alert is observed by DriveView, not by this monitor.
    weak var drowsinessCoordinator: DrowsinessInferenceCoordinator?

    private let cameraManager = CameraManager()

    /// Serializes “Vision busy” so frames are dropped instead of stacking.
    private let processingLock = NSLock()
    private nonisolated(unsafe) var isProcessingFrame = false

    /// Wall-clock start of continuous looking-away; cleared when facing forward or no face.
    private var lookingAwayStartedAt: Date?

    /// Approximate yaw (radians) beyond which the driver is considered looking away.
    private nonisolated static let yawLookAwayThreshold: Double = 0.40

    /// Looking away must persist this long before `attentionState` becomes `.lookingAway`.
    private let lookingAwayAlertDuration: TimeInterval = 2.0

    // MARK: - Lifecycle (legacy single camera)

    /// Starts the front `CameraManager` and routes its frames into Vision.
    /// Permission / configuration failures are published via `cameraError`.
    func start() {
        cameraError = nil

        cameraManager.onFrame = { [weak self] pixelBuffer in
            self?.handleFrame(pixelBuffer)
        }

        cameraManager.requestAccessAndStart { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .success:
                    self.isRunning = true
                    self.cameraError = nil
                case .failure(let error):
                    self.isRunning = false
                    self.cameraError = error
                    self.attentionState = .noFace
                }
            }
        }
    }

    /// Stops accepting frames and shuts down the front camera.
    /// Completion runs on the main queue after the AVCaptureSession has actually stopped.
    /// Does not force-clear `isProcessingFrame` — in-flight Vision work clears it via `defer`.
    func stop(completion: (() -> Void)? = nil) {
        cameraManager.onFrame = nil
        isRunning = false
        lookingAwayStartedAt = nil
        attentionState = .noFace
        cameraManager.stop(completion: completion)
    }

    // MARK: - External frames (MultiCam)

    /// Activates attention processing without starting the legacy single-camera `CameraManager`.
    func beginExternalFrameProcessing() {
        cameraError = nil
        lookingAwayStartedAt = nil
        attentionState = .noFace
        isRunning = true
    }

    /// Ends external-frame mode. Does not force-reset `isProcessingFrame`.
    func endExternalFrameProcessing() {
        isRunning = false
        lookingAwayStartedAt = nil
        attentionState = .noFace
    }

    /// Routes a MultiCam front-camera buffer into the existing Vision path.
    nonisolated func processExternalFrame(_ pixelBuffer: CVPixelBuffer) {
        handleFrame(pixelBuffer)
    }

    // MARK: - Frame intake

    /// Called on the camera frame queue (not the main thread).
    private nonisolated func handleFrame(_ pixelBuffer: CVPixelBuffer) {
        processingLock.lock()
        if isProcessingFrame {
            processingLock.unlock()
            return
        }
        isProcessingFrame = true
        processingLock.unlock()

        analyze(pixelBuffer: pixelBuffer, yawThreshold: Self.yawLookAwayThreshold)
    }

    /// Runs face landmarks + hand pose on one frame, then hops to main to update attention state and
    /// forward a drowsiness feature sample. Hand pose is passed along but ignored by the v3 schema.
    private nonisolated func analyze(pixelBuffer: CVPixelBuffer, yawThreshold: Double) {
        defer {
            processingLock.lock()
            isProcessingFrame = false
            processingLock.unlock()
        }

        // Landmarks request also yields face bounds + pose (yaw/pitch/roll).
        let faceRequest = VNDetectFaceLandmarksRequest()
        let handRequest = VNDetectHumanHandPoseRequest()
        handRequest.maximumHandCount = 2

        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .upMirrored,
            options: [:]
        )

        do {
            try handler.perform([faceRequest, handRequest])
            let faces = faceRequest.results ?? []
            let hands = handRequest.results ?? []

            let instant = Self.classifyInstantState(
                observations: faces,
                yawThreshold: yawThreshold
            )

            let sample = DrowsinessFeatureExtractor.makeSample(
                faces: faces,
                hands: hands,
                timestamp: Date()
            )

            Task { @MainActor in
                self.applyTemporalLogic(instantState: instant)
                self.drowsinessCoordinator?.ingest(sample)
            }

            #if DEBUG
            // Diagnostic only: compare Vision orientations; does not affect production ingest.
            DrowsinessOrientationDebugProbe.considerProbe(pixelBuffer: pixelBuffer)
            #endif
        } catch {
            // Still emit a masked all-zero sample so the temporal stream stays aligned.
            let sample = DrowsinessFeatureExtractor.makeSample(
                faces: [],
                hands: [],
                timestamp: Date()
            )
            Task { @MainActor in
                guard self.isRunning else { return }
                self.lookingAwayStartedAt = nil
                self.attentionState = .noFace
                self.drowsinessCoordinator?.ingest(sample)
            }
        }
    }

    /// Immediate classification from a single frame (before the 2s looking-away gate).
    nonisolated private static func classifyInstantState(
        observations: [VNFaceObservation],
        yawThreshold: Double
    ) -> DriverAttentionState {
        // When multiple faces are visible, provisionally treat the largest face as the driver.
        guard let face = observations.max(by: {
            ($0.boundingBox.width * $0.boundingBox.height) <
            ($1.boundingBox.width * $1.boundingBox.height)
        }) else {
            return .noFace
        }

        // Prefer yaw when Vision provides pose; missing pose → treat as usable but unknown → attentive.
        guard let yawValue = face.yaw?.doubleValue else {
            return .attentive
        }

        if abs(yawValue) >= yawThreshold {
            return .lookingAway
        }
        return .attentive
    }

    // MARK: - Temporal logic

    /// Debounces per-frame states: `.noFace` / `.attentive` apply immediately, while `.lookingAway`
    /// must persist for `lookingAwayAlertDuration` before it is published.
    private func applyTemporalLogic(instantState: DriverAttentionState) {
        // Ignore stale Vision results that finish after stop().
        guard isRunning else { return }

        switch instantState {
        case .noFace:
            lookingAwayStartedAt = nil
            attentionState = .noFace

        case .attentive:
            lookingAwayStartedAt = nil
            attentionState = .attentive

        case .lookingAway:
            let now = Date()
            if lookingAwayStartedAt == nil {
                lookingAwayStartedAt = now
            }

            if let started = lookingAwayStartedAt,
               now.timeIntervalSince(started) >= lookingAwayAlertDuration {
                attentionState = .lookingAway
            } else {
                // Still within grace period — keep showing attentive.
                attentionState = .attentive
            }
        }
    }
}
