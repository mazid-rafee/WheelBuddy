//
//  MultiCamManager.swift
//  WheelBuddy
//

import AVFoundation
import Foundation

enum MultiCamError: LocalizedError, Equatable {
    case permissionDenied
    case permissionRestricted
    case multiCamUnsupported
    case frontCameraUnavailable
    case rearCameraUnavailable
    case unsupportedCameraPair
    case noSupportedFrontFormat
    case noSupportedRearFormat
    case cannotCreateInput(String)
    case cannotAddInput
    case cannotAddOutput
    case cannotAddConnection
    case hardwareCostTooHigh(Float)
    case configurationFailed(String)

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            return "Camera access was denied. Enable it in Settings."
        case .permissionRestricted:
            return "Camera access is restricted on this device."
        case .multiCamUnsupported:
            return "Multi-camera capture is not supported on this device."
        case .frontCameraUnavailable:
            return "Front camera is unavailable."
        case .rearCameraUnavailable:
            return "Rear camera is unavailable."
        case .unsupportedCameraPair:
            return "Front + rear wide cameras are not a supported multi-cam pair on this device."
        case .noSupportedFrontFormat:
            return "No suitable MultiCam-supported front camera format was found."
        case .noSupportedRearFormat:
            return "No suitable MultiCam-supported rear camera format was found."
        case .cannotCreateInput(let message):
            return "Could not create camera input: \(message)"
        case .cannotAddInput:
            return "Could not add a camera input to the multi-cam session."
        case .cannotAddOutput:
            return "Could not add a video output to the multi-cam session."
        case .cannotAddConnection:
            return "Could not connect a camera input to its video output."
        case .hardwareCostTooHigh(let cost):
            return String(format: "MultiCam hardwareCost is too high (%.2f > 1.0).", cost)
        case .configurationFailed(let message):
            return "MultiCam configuration failed: \(message)"
        }
    }
}

/// Owns a single `AVCaptureMultiCamSession` and routes front/rear frames to separate callbacks.
/// Does not run Vision or Core ML — only capture + delivery.
nonisolated final class MultiCamManager: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private let session = AVCaptureMultiCamSession()
    private let sessionQueue = DispatchQueue(label: "com.wheelbuddy.multicam.session")
    private let frontOutputQueue = DispatchQueue(label: "com.wheelbuddy.multicam.front.frames", qos: .userInitiated)
    private let rearOutputQueue = DispatchQueue(label: "com.wheelbuddy.multicam.rear.frames", qos: .userInitiated)

    private let frontVideoOutput = AVCaptureVideoDataOutput()
    private let rearVideoOutput = AVCaptureVideoDataOutput()

    private var isConfigured = false
    private var isDeliveringFrames = false
    private var observersRegistered = false

    /// Called on `frontOutputQueue`.
    var onFrontFrame: ((CVPixelBuffer) -> Void)?
    /// Called on `rearOutputQueue`.
    var onRearFrame: ((CVPixelBuffer) -> Void)?

    var isRunning: Bool {
        session.isRunning
    }

    private let targetFrontFPS: Double = 15
    private let targetRearFPS: Double = 15

    func requestAccessAndStart(completion: @escaping (Result<Void, MultiCamError>) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndStart(completion: completion)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                guard let self else { return }
                if granted {
                    self.configureAndStart(completion: completion)
                } else {
                    DispatchQueue.main.async {
                        completion(.failure(.permissionDenied))
                    }
                }
            }
        case .denied:
            DispatchQueue.main.async {
                completion(.failure(.permissionDenied))
            }
        case .restricted:
            DispatchQueue.main.async {
                completion(.failure(.permissionRestricted))
            }
        @unknown default:
            DispatchQueue.main.async {
                completion(.failure(.permissionDenied))
            }
        }
    }

    /// Stops the multi-cam session on `sessionQueue`. Completion runs on the main queue
    /// only after `stopRunning()` has returned (or immediately if already stopped).
    func stop(completion: (() -> Void)? = nil) {
        sessionQueue.async { [weak self] in
            guard let self else {
                if let completion {
                    DispatchQueue.main.async(execute: completion)
                }
                return
            }

            self.isDeliveringFrames = false
            self.onFrontFrame = nil
            self.onRearFrame = nil

            if self.session.isRunning {
                self.session.stopRunning()
            }

            print("[MultiCam] Session stopped")

            if let completion {
                DispatchQueue.main.async(execute: completion)
            }
        }
    }

    // MARK: - Configure + start

    private func configureAndStart(completion: @escaping (Result<Void, MultiCamError>) -> Void) {
        sessionQueue.async { [weak self] in
            guard let self else { return }

            do {
                if !self.isConfigured {
                    try self.configureSession()
                    self.isConfigured = true
                }

                self.isDeliveringFrames = true

                if !self.session.isRunning {
                    self.session.startRunning()
                }

                print("[MultiCam] Session started")
                DispatchQueue.main.async {
                    completion(.success(()))
                }
            } catch let error as MultiCamError {
                self.isDeliveringFrames = false
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            } catch {
                self.isDeliveringFrames = false
                DispatchQueue.main.async {
                    completion(.failure(.configurationFailed(error.localizedDescription)))
                }
            }
        }
    }

    private func configureSession() throws {
        let supported = AVCaptureMultiCamSession.isMultiCamSupported
        print("[MultiCam] Supported: \(supported)")
        guard supported else {
            throw MultiCamError.multiCamUnsupported
        }

        registerSessionObserversIfNeeded()

        guard let frontDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            throw MultiCamError.frontCameraUnavailable
        }
        guard let rearDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw MultiCamError.rearCameraUnavailable
        }

        try assertSupportedMultiCamPair(front: frontDevice, rear: rearDevice)

        let frontFormat = try Self.selectFormat(
            for: frontDevice,
            targetWidth: 640,
            targetHeight: 480,
            targetFPS: targetFrontFPS,
            role: "front"
        )
        let rearFormat = try Self.selectFormat(
            for: rearDevice,
            targetWidth: 1280,
            targetHeight: 720,
            targetFPS: targetRearFPS,
            role: "rear"
        )

        try Self.applyFormat(frontFormat, on: frontDevice, targetFPS: targetFrontFPS)
        try Self.applyFormat(rearFormat, on: rearDevice, targetFPS: targetRearFPS)

        let frontInput: AVCaptureDeviceInput
        let rearInput: AVCaptureDeviceInput
        do {
            frontInput = try AVCaptureDeviceInput(device: frontDevice)
            rearInput = try AVCaptureDeviceInput(device: rearDevice)
        } catch {
            throw MultiCamError.cannotCreateInput(error.localizedDescription)
        }

        // Frame rate is locked via activeVideoMin/MaxFrameDuration on each device format.
        // (videoMinFrameDurationOverride availability varies; device-level duration is sufficient.)

        session.beginConfiguration()
        defer { session.commitConfiguration() }

        // Explicit MultiCam graph: no automatic connections.
        guard session.canAddInput(frontInput) else { throw MultiCamError.cannotAddInput }
        session.addInputWithNoConnections(frontInput)

        guard session.canAddInput(rearInput) else { throw MultiCamError.cannotAddInput }
        session.addInputWithNoConnections(rearInput)

        configureVideoOutput(frontVideoOutput, queue: frontOutputQueue)
        configureVideoOutput(rearVideoOutput, queue: rearOutputQueue)

        guard session.canAddOutput(frontVideoOutput) else { throw MultiCamError.cannotAddOutput }
        session.addOutputWithNoConnections(frontVideoOutput)

        guard session.canAddOutput(rearVideoOutput) else { throw MultiCamError.cannotAddOutput }
        session.addOutputWithNoConnections(rearVideoOutput)

        guard let frontPort = Self.videoPort(from: frontInput, device: frontDevice) else {
            throw MultiCamError.configurationFailed("Front video port unavailable.")
        }
        guard let rearPort = Self.videoPort(from: rearInput, device: rearDevice) else {
            throw MultiCamError.configurationFailed("Rear video port unavailable.")
        }

        let frontConnection = AVCaptureConnection(inputPorts: [frontPort], output: frontVideoOutput)
        guard session.canAddConnection(frontConnection) else { throw MultiCamError.cannotAddConnection }
        session.addConnection(frontConnection)
        configureConnection(frontConnection, mirror: true)

        let rearConnection = AVCaptureConnection(inputPorts: [rearPort], output: rearVideoOutput)
        guard session.canAddConnection(rearConnection) else { throw MultiCamError.cannotAddConnection }
        session.addConnection(rearConnection)
        configureConnection(rearConnection, mirror: false)

        let hardwareCost = session.hardwareCost
        let systemPressureCost = session.systemPressureCost
        print(String(format: "[MultiCam] hardwareCost: %.2f", hardwareCost))
        print(String(format: "[MultiCam] systemPressureCost: %.2f", systemPressureCost))

        if hardwareCost > 1.0 {
            throw MultiCamError.hardwareCostTooHigh(hardwareCost)
        }
        if systemPressureCost > 1.0 {
            print(String(format: "[MultiCam] WARNING: systemPressureCost %.2f > 1.0", systemPressureCost))
        }

        Self.logSelectedFormat(frontFormat, role: "Front", fps: targetFrontFPS)
        Self.logSelectedFormat(rearFormat, role: "Rear", fps: targetRearFPS)
        print("[MultiCam] Graph: front wide → frontVideoOutput; rear wide → rearVideoOutput (manual connections)")
    }

    private func configureVideoOutput(_ output: AVCaptureVideoDataOutput, queue: DispatchQueue) {
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: queue)
    }

    /// Preserve Milestone 1/2.1 delivered-buffer semantics:
    /// 90° rotation for portrait; mirror front only. DriverMonitor uses `.upMirrored`;
    /// RoadDetectionService still uses `.up` for the already-rotated rear buffer.
    private func configureConnection(_ connection: AVCaptureConnection, mirror: Bool) {
        if connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
        if connection.isVideoMirroringSupported {
            connection.isVideoMirrored = mirror
        }
    }

    private func assertSupportedMultiCamPair(front: AVCaptureDevice, rear: AVCaptureDevice) throws {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video,
            position: .unspecified
        )
        let sets = discovery.supportedMultiCamDeviceSets
        let pairSupported = sets.contains { set in
            set.contains(front) && set.contains(rear)
        }
        print("[MultiCam] supportedMultiCamDeviceSets contains front+rear wide: \(pairSupported) (sets: \(sets.count))")
        guard pairSupported else {
            throw MultiCamError.unsupportedCameraPair
        }
    }

    private static func videoPort(from input: AVCaptureDeviceInput, device: AVCaptureDevice) -> AVCaptureInput.Port? {
        if let port = input.ports(
            for: .video,
            sourceDeviceType: device.deviceType,
            sourceDevicePosition: device.position
        ).first {
            return port
        }
        return input.ports.first { $0.mediaType == .video }
    }

    // MARK: - Format selection

    private static func selectFormat(
        for device: AVCaptureDevice,
        targetWidth: Int32,
        targetHeight: Int32,
        targetFPS: Double,
        role: String
    ) throws -> AVCaptureDevice.Format {
        let candidates = device.formats.filter { $0.isMultiCamSupported }
        guard !candidates.isEmpty else {
            throw role == "front" ? MultiCamError.noSupportedFrontFormat : MultiCamError.noSupportedRearFormat
        }

        func dimensions(of format: AVCaptureDevice.Format) -> CMVideoDimensions {
            CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        }

        func maxFPS(of format: AVCaptureDevice.Format) -> Double {
            format.videoSupportedFrameRateRanges.map { $0.maxFrameRate }.max() ?? 0
        }

        func supportsFPS(_ format: AVCaptureDevice.Format, fps: Double) -> Bool {
            format.videoSupportedFrameRateRanges.contains { range in
                range.minFrameRate <= fps && fps <= range.maxFrameRate
            }
        }

        let targetArea = Int(targetWidth) * Int(targetHeight)

        let scored = candidates
            .filter { supportsFPS($0, fps: targetFPS) || maxFPS(of: $0) >= targetFPS }
            .map { format -> (AVCaptureDevice.Format, Int) in
                let dim = dimensions(of: format)
                let area = Int(dim.width) * Int(dim.height)
                let areaDelta = abs(area - targetArea)
                // Prefer formats that explicitly include target FPS.
                let fpsPenalty = supportsFPS(format, fps: targetFPS) ? 0 : 50_000_000
                // Soft-penalize very large formats (4K etc.).
                let hugePenalty = area > 2_500_000 ? area : 0
                return (format, areaDelta + fpsPenalty + hugePenalty)
            }
            .sorted { $0.1 < $1.1 }

        guard let best = scored.first?.0 else {
            throw role == "front" ? MultiCamError.noSupportedFrontFormat : MultiCamError.noSupportedRearFormat
        }
        return best
    }

    private static func applyFormat(_ format: AVCaptureDevice.Format, on device: AVCaptureDevice, targetFPS: Double) throws {
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }

        device.activeFormat = format
        let duration = CMTime(value: 1, timescale: CMTimeScale(targetFPS))
        if format.videoSupportedFrameRateRanges.contains(where: { $0.minFrameRate <= targetFPS && targetFPS <= $0.maxFrameRate }) {
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration
        } else if let range = format.videoSupportedFrameRateRanges.max(by: { $0.maxFrameRate < $1.maxFrameRate }) {
            let clamped = min(max(targetFPS, range.minFrameRate), range.maxFrameRate)
            let clampedDuration = CMTime(value: 1, timescale: CMTimeScale(clamped))
            device.activeVideoMinFrameDuration = clampedDuration
            device.activeVideoMaxFrameDuration = clampedDuration
        }
    }

    private static func logSelectedFormat(_ format: AVCaptureDevice.Format, role: String, fps: Double) {
        let dim = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        print("[MultiCam] \(role) format: \(dim.width)x\(dim.height) @ \(Int(fps)) fps")
    }

    // MARK: - Notifications

    private func registerSessionObserversIfNeeded() {
        guard !observersRegistered else { return }
        observersRegistered = true

        let center = NotificationCenter.default
        center.addObserver(
            forName: AVCaptureSession.runtimeErrorNotification,
            object: session,
            queue: nil
        ) { note in
            let error = note.userInfo?[AVCaptureSessionErrorKey] as? AVError
            print("[MultiCam] Runtime error: \(error?.localizedDescription ?? "unknown")")
        }

        center.addObserver(
            forName: AVCaptureSession.wasInterruptedNotification,
            object: session,
            queue: nil
        ) { note in
            let raw = note.userInfo?[AVCaptureSessionInterruptionReasonKey] as? Int
            print("[MultiCam] Session interrupted: reason=\(raw.map(String.init) ?? "unknown")")
        }

        center.addObserver(
            forName: AVCaptureSession.interruptionEndedNotification,
            object: session,
            queue: nil
        ) { _ in
            print("[MultiCam] Session interruption ended")
        }
    }

    // MARK: - Frame delivery

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard isDeliveringFrames else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        if output === frontVideoOutput {
            #if DEBUG
            DrowsinessOrientationDebugProbe.updateConnection(
                rotationDegrees: connection.videoRotationAngle,
                mirrored: connection.isVideoMirrored
            )
            #endif
            onFrontFrame?(pixelBuffer)
        } else if output === rearVideoOutput {
            onRearFrame?(pixelBuffer)
        }
    }
}
