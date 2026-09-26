//
//  DrowsinessOrientationDebugProbe.swift
//  WheelBuddy
//
//  DEBUG-only camera orientation A/B probe for front-camera Vision inputs.
//  Does not affect production inference, API packets, or UI.
//

#if DEBUG
import CoreVideo
import Foundation
import Vision

/// Temporary front-camera orientation diagnostic.
///
/// Enable/disable with ``isEnabled`` (default `true` in DEBUG builds).
/// Automatically stops after ``probeDurationSeconds`` from the first sample.
enum DrowsinessOrientationDebugProbe {
    /// Flip to `false` to silence the probe without removing the call site.
    static var isEnabled: Bool = true

    private static let probeDurationSeconds: TimeInterval = 20
    private static let probeIntervalSeconds: TimeInterval = 1

    private static let lock = NSLock()
    private static var startedAt: Date?
    private static var lastProbeAt: Date?
    private static var finished = false

    /// Latest front `AVCaptureConnection` values (updated from MultiCam).
    private static var connectionRotation: Double = 90
    private static var connectionMirrored: Bool = true

    private static let orientations: [(name: String, value: CGImagePropertyOrientation)] = [
        ("upMirrored", .upMirrored),
        ("leftMirrored", .leftMirrored),
        ("rightMirrored", .rightMirrored),
    ]

    static func updateConnection(rotationDegrees: Double, mirrored: Bool) {
        lock.lock()
        connectionRotation = rotationDegrees
        connectionMirrored = mirrored
        lock.unlock()
    }

    /// Run at most once per second for 20s. Never ingests into the inference coordinator.
    static func considerProbe(pixelBuffer: CVPixelBuffer) {
        guard isEnabled else { return }

        let now = Date()
        lock.lock()
        if finished {
            lock.unlock()
            return
        }
        if startedAt == nil {
            startedAt = now
            print(
                "[OrientationTest] starting 20s probe "
                    + "(upMirrored / leftMirrored / rightMirrored); "
                    + "production pipeline unchanged"
            )
        }
        if let started = startedAt, now.timeIntervalSince(started) >= probeDurationSeconds {
            finished = true
            lock.unlock()
            print("[OrientationTest] finished after \(Int(probeDurationSeconds))s")
            return
        }
        if let last = lastProbeAt, now.timeIntervalSince(last) < probeIntervalSeconds {
            lock.unlock()
            return
        }
        lastProbeAt = now
        let rotation = connectionRotation
        let mirrored = connectionMirrored
        lock.unlock()

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        for entry in orientations {
            let faceRequest = VNDetectFaceLandmarksRequest()
            let handler = VNImageRequestHandler(
                cvPixelBuffer: pixelBuffer,
                orientation: entry.value,
                options: [:]
            )
            do {
                try handler.perform([faceRequest])
            } catch {
                print(
                    "[OrientationTest] orientation=\(entry.name) "
                        + "buffer_width=\(width) buffer_height=\(height) "
                        + "connection_rotation=\(format(rotation)) "
                        + "connection_mirrored=\(mirrored) "
                        + "yaw=nan pitch=nan roll=nan left_EAR=nan right_EAR=nan "
                        + "error=\(error.localizedDescription)"
                )
                continue
            }

            let faces = faceRequest.results ?? []
            // Diagnostic only — never send to the inference server.
            let sample = DrowsinessFeatureExtractor.makeSample(
                faces: faces,
                hands: [],
                timestamp: now
            )
            let values = sample.values
            let yaw = values.indices.contains(1) ? values[1] : 0
            let pitch = values.indices.contains(2) ? values[2] : 0
            let roll = values.indices.contains(3) ? values[3] : 0
            let leftEAR = values.indices.contains(6) ? values[6] : 0
            let rightEAR = values.indices.contains(7) ? values[7] : 0

            print(
                "[OrientationTest] "
                    + "orientation=\(entry.name) "
                    + "buffer_width=\(width) "
                    + "buffer_height=\(height) "
                    + "connection_rotation=\(format(rotation)) "
                    + "connection_mirrored=\(mirrored) "
                    + "yaw=\(format(yaw)) "
                    + "pitch=\(format(pitch)) "
                    + "roll=\(format(roll)) "
                    + "left_EAR=\(format(leftEAR)) "
                    + "right_EAR=\(format(rightEAR))"
            )
        }
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.4f", value)
    }
}
#endif
