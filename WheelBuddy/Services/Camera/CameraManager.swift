//
//  CameraManager.swift
//  WheelBuddy
//

import AVFoundation
import Foundation

/// User-presentable camera setup failures (permission or session configuration).
enum CameraError: LocalizedError, Equatable {
    case permissionDenied
    case permissionRestricted
    case cameraUnavailable
    case cannotAddInput
    case cannotAddOutput
    case configurationFailed(String)

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            return "Camera access was denied. Enable it in Settings."
        case .permissionRestricted:
            return "Camera access is restricted on this device."
        case .cameraUnavailable:
            return "The requested camera is unavailable."
        case .cannotAddInput:
            return "Could not configure the camera input."
        case .cannotAddOutput:
            return "Could not configure camera video output."
        case .configurationFailed(let message):
            return "Camera configuration failed: \(message)"
        }
    }
}

/// Owns a single AVCaptureSession and delivers frames off the main thread.
/// Defaults to the front camera so Milestone 1 behavior is unchanged.
///
/// `@unchecked Sendable`: all session configuration and start/stop happen on the serial
/// `sessionQueue`; frames are delivered on `videoOutputQueue`.
nonisolated final class CameraManager: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private let position: AVCaptureDevice.Position
    private let session = AVCaptureSession()
    /// Serial queue for all blocking AVCaptureSession work (configure / start / stop).
    private let sessionQueue = DispatchQueue(label: "com.wheelbuddy.camera.session")
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoOutputQueue = DispatchQueue(label: "com.wheelbuddy.camera.frames", qos: .userInitiated)

    /// Only accessed on `sessionQueue`; the session is configured once and reused across restarts.
    private var isConfigured = false

    /// Called on `videoOutputQueue` with each delivered pixel buffer.
    var onFrame: ((CVPixelBuffer) -> Void)?

    var isRunning: Bool {
        session.isRunning
    }

    /// - Parameter position: `.front` (default) for driver monitoring; `.back` for road detection.
    init(position: AVCaptureDevice.Position = .front) {
        self.position = position
        super.init()
    }

    // MARK: - Lifecycle

    /// Requests camera permission if needed, then configures (first time only) and starts the session.
    /// Completion runs on the main queue. If the manager is deallocated before configuration runs
    /// (e.g. while the permission prompt is showing), completion is never called.
    func requestAccessAndStart(completion: @escaping (Result<Void, CameraError>) -> Void) {
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

    /// Stops the capture session on `sessionQueue`. Completion runs on the main queue
    /// only after `stopRunning()` has returned (or immediately if already stopped).
    func stop(completion: (() -> Void)? = nil) {
        sessionQueue.async { [weak self] in
            guard let self else {
                if let completion {
                    DispatchQueue.main.async(execute: completion)
                }
                return
            }

            if self.session.isRunning {
                self.session.stopRunning()
            }

            if let completion {
                DispatchQueue.main.async(execute: completion)
            }
        }
    }

    // MARK: - Configuration

    private func configureAndStart(completion: @escaping (Result<Void, CameraError>) -> Void) {
        sessionQueue.async { [weak self] in
            guard let self else { return }

            do {
                if !self.isConfigured {
                    try self.configureSession()
                    self.isConfigured = true
                }

                if !self.session.isRunning {
                    self.session.startRunning()
                }

                DispatchQueue.main.async {
                    completion(.success(()))
                }
            } catch let error as CameraError {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(.configurationFailed(error.localizedDescription)))
                }
            }
        }
    }

    /// Must run on `sessionQueue`. Wide-angle camera at `.medium` preset, 32BGRA output,
    /// late frames discarded, rotated 90° to portrait where supported.
    private func configureSession() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .medium

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) else {
            throw CameraError.cameraUnavailable
        }

        let input: AVCaptureDeviceInput
        do {
            input = try AVCaptureDeviceInput(device: device)
        } catch {
            throw CameraError.configurationFailed(error.localizedDescription)
        }

        guard session.canAddInput(input) else {
            throw CameraError.cannotAddInput
        }
        session.addInput(input)

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.setSampleBufferDelegate(self, queue: videoOutputQueue)

        guard session.canAddOutput(videoOutput) else {
            throw CameraError.cannotAddOutput
        }
        session.addOutput(videoOutput)

        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
            // Mirror only the front camera (driver-facing preview semantics).
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = (position == .front)
            }
        }
    }

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(pixelBuffer)
    }
}
