//
//  RoadDetectionService.swift
//  WheelBuddy
//

import AVFoundation
import Combine
import CoreML
import Foundation
import Vision

/// Rear-camera road object detection via the bundled `RoadObjectDetector` Core ML model + Vision.
/// Named separately from the Xcode-generated `RoadObjectDetector` model class to avoid a symbol clash.
/// Owns its own back-camera `CameraManager`. Only one camera session should run at a time app-wide.
@MainActor
final class RoadDetectionService: ObservableObject {
    @Published private(set) var detections: [RoadDetection] = []
    @Published private(set) var state: RoadMonitoringState = .clear
    @Published private(set) var cameraError: CameraError?
    @Published private(set) var isRunning = false
    @Published private(set) var isModelReady = false
    @Published private(set) var modelLoadMessage: String = "Looking for RoadObjectDetector…"

    /// Easy-to-tune minimum confidence for retained detections.
    nonisolated static let confidenceThreshold: Float = 0.35

    /// Roughly 5 inferences/sec.
    nonisolated static let minimumInferenceInterval: TimeInterval = 0.2

    /// Log average inference time every N completed inferences.
    nonisolated static let inferenceTimingSampleCount: Int = 20

    /// COCO-style labels we care about for road monitoring (easy to extend later).
    nonisolated static let relevantLabels: Set<String> = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck"
    ]

    private let cameraManager = CameraManager(position: .back)

    private let processingLock = NSLock()
    private nonisolated(unsafe) var isProcessingFrame = false
    private nonisolated(unsafe) var lastInferenceAt: CFAbsoluteTime = 0

    /// Loaded once and reused. Vision request is also reused.
    private nonisolated(unsafe) var visionModel: VNCoreMLModel?
    private nonisolated(unsafe) var detectionRequest: VNCoreMLRequest?

    /// Log unexpected Vision result types once per service instance.
    private nonisolated(unsafe) var hasLoggedUnexpectedResultType = false

    /// Rolling inference timing (updated only on the processing path).
    private nonisolated(unsafe) var inferenceTimingCount = 0
    private nonisolated(unsafe) var inferenceTimingAccumulatedMs: Double = 0

    /// CameraManager already rotates video output by 90° for portrait, and the rear
    /// camera is not mirrored. The delivered CVPixelBuffer is therefore upright → `.up`.
    /// (DriverMonitor uses `.upMirrored` for the front path — do not change that here.
    /// MultiCam later may centralize/revisit orientation handling.)
    private nonisolated static let visionOrientation: CGImagePropertyOrientation = .up

    init() {
        loadModel()
    }

    func start() {
        cameraError = nil

        guard isModelReady else {
            state = .modelUnavailable
            print("[RoadDetector] Cannot start — model unavailable.")
            return
        }

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
                    print("[RoadDetector] Rear camera started.")
                case .failure(let error):
                    self.isRunning = false
                    self.cameraError = error
                    self.detections = []
                    self.state = self.isModelReady ? .clear : .modelUnavailable
                    print("[RoadDetector] Camera start failed: \(error.localizedDescription)")
                }
            }
        }
    }

    /// Stops accepting frames and shuts down the rear camera.
    /// Completion runs on the main queue after the AVCaptureSession has actually stopped.
    /// Does not force-clear `isProcessingFrame` — in-flight inference clears it via `defer`.
    func stop(completion: (() -> Void)? = nil) {
        cameraManager.onFrame = nil
        isRunning = false
        detections = []
        if isModelReady {
            state = .clear
        }
        print("[RoadDetector] Stopping rear camera…")
        cameraManager.stop { [weak self] in
            print("[RoadDetector] Rear camera stopped.")
            completion?()
            _ = self
        }
    }

    // MARK: - External frames (MultiCam)

    /// Activates road detection without starting the legacy single-camera `CameraManager`.
    /// Model-unavailable leaves driver monitoring unaffected.
    func beginExternalFrameProcessing() {
        cameraError = nil
        detections = []

        guard isModelReady else {
            state = .modelUnavailable
            isRunning = false
            print("[RoadDetector] External mode — model unavailable (driver path unaffected).")
            return
        }

        state = .clear
        isRunning = true
        print("[RoadDetector] External frame processing active.")
    }

    /// Ends external-frame mode. Does not force-reset `isProcessingFrame`.
    func endExternalFrameProcessing() {
        isRunning = false
        detections = []
        if isModelReady {
            state = .clear
        }
    }

    /// Routes a MultiCam rear-camera buffer into the existing Core ML path.
    nonisolated func processExternalFrame(_ pixelBuffer: CVPixelBuffer) {
        handleFrame(pixelBuffer)
    }

    // MARK: - Model loading

    /// Loads the bundled `RoadObjectDetector` Core ML model once.
    private func loadModel() {
        do {
            let mlModel = try Self.makeMLModel()
            let vnModel = try VNCoreMLModel(for: mlModel)
            let request = VNCoreMLRequest(model: vnModel)
            request.imageCropAndScaleOption = .scaleFill

            visionModel = vnModel
            detectionRequest = request
            isModelReady = true
            state = .clear
            modelLoadMessage = "RoadObjectDetector loaded"
            print("[RoadDetector] Model loaded successfully.")
        } catch {
            visionModel = nil
            detectionRequest = nil
            isModelReady = false
            state = .modelUnavailable
            modelLoadMessage = error.localizedDescription
            print("[RoadDetector] Model load failed: \(error.localizedDescription)")
            Self.logBundleModelCandidates()
        }
    }

    nonisolated private static func makeMLModel() throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = .all

        // Xcode compiles WheelBuddy/ML/RoadObjectDetector.mlpackage → RoadObjectDetector.mlmodelc in the app bundle.
        let resourceNames = ["RoadObjectDetector", "yolov8n", "YOLOv8n"]
        for name in resourceNames {
            if let modelURL = Bundle.main.url(forResource: name, withExtension: "mlmodelc")
                ?? Bundle.main.url(forResource: name, withExtension: "mlmodelc", subdirectory: "ML") {
                print("[RoadDetector] Loading compiled model at \(modelURL.path)")
                return try MLModel(contentsOf: modelURL, configuration: config)
            }
        }

        for name in resourceNames {
            if let packageURL = Bundle.main.url(forResource: name, withExtension: "mlpackage")
                ?? Bundle.main.url(forResource: name, withExtension: "mlpackage", subdirectory: "ML") {
                print("[RoadDetector] Compiling mlpackage at \(packageURL.path)")
                let compiledURL = try MLModel.compileModel(at: packageURL)
                return try MLModel(contentsOf: compiledURL, configuration: config)
            }

            if let mlmodelURL = Bundle.main.url(forResource: name, withExtension: "mlmodel")
                ?? Bundle.main.url(forResource: name, withExtension: "mlmodel", subdirectory: "ML") {
                print("[RoadDetector] Compiling mlmodel at \(mlmodelURL.path)")
                let compiledURL = try MLModel.compileModel(at: mlmodelURL)
                return try MLModel(contentsOf: compiledURL, configuration: config)
            }
        }

        if let all = Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil) {
            if let match = all.first(where: {
                let n = $0.deletingPathExtension().lastPathComponent.lowercased()
                return n.contains("road") || n.contains("yolo") || n.contains("object")
            }) {
                print("[RoadDetector] Loading discovered model at \(match.path)")
                return try MLModel(contentsOf: match, configuration: config)
            }
        }

        throw RoadDetectorError.modelNotFound
    }

    nonisolated private static func logBundleModelCandidates() {
        let mlmodelc = Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil) ?? []
        let mlpackage = Bundle.main.urls(forResourcesWithExtension: "mlpackage", subdirectory: nil) ?? []
        let mlmodel = Bundle.main.urls(forResourcesWithExtension: "mlmodel", subdirectory: nil) ?? []
        print("[RoadDetector] Bundle mlmodelc: \(mlmodelc.map(\.lastPathComponent))")
        print("[RoadDetector] Bundle mlpackage: \(mlpackage.map(\.lastPathComponent))")
        print("[RoadDetector] Bundle mlmodel: \(mlmodel.map(\.lastPathComponent))")
    }

    // MARK: - Frame intake

    private nonisolated func handleFrame(_ pixelBuffer: CVPixelBuffer) {
        let now = CFAbsoluteTimeGetCurrent()

        processingLock.lock()
        if isProcessingFrame || (now - lastInferenceAt) < Self.minimumInferenceInterval {
            processingLock.unlock()
            return
        }
        isProcessingFrame = true
        lastInferenceAt = now
        processingLock.unlock()

        analyze(pixelBuffer: pixelBuffer)
    }

    private nonisolated func analyze(pixelBuffer: CVPixelBuffer) {
        defer {
            processingLock.lock()
            isProcessingFrame = false
            processingLock.unlock()
        }

        guard let request = detectionRequest else {
            Task { @MainActor in
                guard self.isRunning else { return }
                self.state = .modelUnavailable
                self.detections = []
            }
            return
        }

        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: Self.visionOrientation,
            options: [:]
        )

        do {
            let start = CFAbsoluteTimeGetCurrent()
            try handler.perform([request])
            let elapsedMs = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
            recordInferenceTiming(elapsedMs: elapsedMs)

            let filtered: [RoadDetection]
            if let observations = request.results as? [VNRecognizedObjectObservation] {
                filtered = Self.parseDetections(from: observations)
            } else if let results = request.results, !results.isEmpty {
                // Non-empty but not VNRecognizedObjectObservation — incompatible model output.
                logUnexpectedResultTypesIfNeeded(results)
                filtered = []
            } else {
                // nil/empty results: treat as no detections (distinct from wrong type).
                filtered = []
            }

            Task { @MainActor in
                self.publish(detections: filtered)
            }
        } catch {
            print("[RoadDetector] Vision inference failed: \(error.localizedDescription)")
            Task { @MainActor in
                guard self.isRunning else { return }
                if self.detections.isEmpty {
                    self.state = .clear
                }
            }
        }
    }

    nonisolated private func recordInferenceTiming(elapsedMs: Double) {
        processingLock.lock()
        inferenceTimingAccumulatedMs += elapsedMs
        inferenceTimingCount += 1
        let count = inferenceTimingCount
        let total = inferenceTimingAccumulatedMs
        if count >= Self.inferenceTimingSampleCount {
            inferenceTimingCount = 0
            inferenceTimingAccumulatedMs = 0
            processingLock.unlock()
            let average = total / Double(Self.inferenceTimingSampleCount)
            print(String(format: "[RoadDetector] Avg inference (%d): %.1f ms", Self.inferenceTimingSampleCount, average))
        } else {
            processingLock.unlock()
        }
    }

    nonisolated private func logUnexpectedResultTypesIfNeeded(_ results: [Any]) {
        processingLock.lock()
        let alreadyLogged = hasLoggedUnexpectedResultType
        if !alreadyLogged {
            hasLoggedUnexpectedResultType = true
        }
        processingLock.unlock()

        guard !alreadyLogged else { return }

        let typeNames = results.map { String(describing: type(of: $0)) }
        let unique = Array(Set(typeNames)).sorted()
        print("[RoadDetector] Unexpected Vision result type: \(unique.joined(separator: ", "))")
    }

    nonisolated private static func parseDetections(
        from observations: [VNRecognizedObjectObservation]
    ) -> [RoadDetection] {
        var results: [RoadDetection] = []

        for observation in observations {
            guard let top = observation.labels.first else { continue }
            let label = top.identifier.lowercased()
            // Combined objectness × class confidence.
            let confidence = observation.confidence * top.confidence

            guard relevantLabels.contains(label) else { continue }
            guard confidence >= confidenceThreshold else { continue }

            results.append(
                RoadDetection(
                    label: label,
                    confidence: confidence,
                    boundingBox: observation.boundingBox
                )
            )
        }

        results.sort { $0.confidence > $1.confidence }
        return results
    }

    private func publish(detections: [RoadDetection]) {
        // Ignore stale inference that finishes after stop().
        guard isRunning else { return }

        self.detections = detections

        if !isModelReady {
            state = .modelUnavailable
            return
        }

        if detections.isEmpty {
            state = .clear
        } else {
            state = .objectsDetected
            for detection in detections.prefix(5) {
                print(String(format: "[RoadDetector] %@ %.2f", detection.label, detection.confidence))
            }
            print("[RoadDetector] \(detections.count) relevant objects")
        }
    }
}

private enum RoadDetectorError: LocalizedError {
    case modelNotFound

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "RoadObjectDetector model was not found in the app bundle. Add WheelBuddy/ML/RoadObjectDetector.mlpackage to the target and rebuild."
        }
    }
}
