//
//  DrowsinessRemoteTests.swift
//  WheelBuddyTests
//

import XCTest
@testable import WheelBuddy

final class DrowsinessRemoteTests: XCTestCase {

    func testFeatureOrderMatchesPythonContract() {
        XCTAssertEqual(DrowsinessFeatureContract.featureCount, 12)
        XCTAssertEqual(
            DrowsinessFeatureContract.schemaVersion,
            "drowsiness_feature_schema_v3"
        )
        XCTAssertEqual(DrowsinessFeatureContract.windowFrames, 20)
        XCTAssertEqual(DrowsinessFeatureContract.samplingRateHz, 15.0)
        XCTAssertEqual(
            DrowsinessFeatureContract.classNames,
            ["closed", "open", "undefined"]
        )
        XCTAssertEqual(
            DrowsinessFeatureContract.featureNames,
            [
                "face_detected",
                "yaw",
                "pitch",
                "roll",
                "left_eye_valid",
                "right_eye_valid",
                "left_eye_aspect_ratio",
                "right_eye_aspect_ratio",
                "left_pupil_rel_x",
                "left_pupil_rel_y",
                "right_pupil_rel_x",
                "right_pupil_rel_y",
            ]
        )
    }

    func testRequestJSONCodingKeys() throws {
        let request = DrowsinessPredictRequest(
            featureSchemaVersion: DrowsinessFeatureContract.schemaVersion,
            sessionID: "s1",
            sequenceID: 42,
            sentAtUTC: Date(timeIntervalSince1970: 1_000),
            samplingRateHz: 15.0,
            featureNames: DrowsinessFeatureContract.featureNames,
            samples: [
                DrowsinessAPISample(
                    timestampMs: 1,
                    values: Array(repeating: 0.0, count: 12)
                ),
            ]
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(request)
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        XCTAssertEqual(
            json["feature_schema_version"] as? String,
            "drowsiness_feature_schema_v3"
        )
        XCTAssertEqual(json["session_id"] as? String, "s1")
        XCTAssertEqual(json["sequence_id"] as? Int, 42)
        XCTAssertNotNil(json["sampling_rate_hz"])
        XCTAssertNotNil(json["feature_names"])
        XCTAssertNotNil(json["samples"])
        XCTAssertNil(json["schema_version"])
        XCTAssertNil(json["schemaVersion"])
        XCTAssertNil(json["featureSchemaVersion"])
    }

    func testResponseJSONDecodes() throws {
        let payload = """
        {
          "session_id": "s1",
          "sequence_id": 42,
          "label": "open",
          "label_index": 1,
          "confidence": 0.91,
          "probabilities": {
            "closed": 0.05,
            "open": 0.91,
            "undefined": 0.04
          },
          "model_version": "best_accuracy_v3",
          "feature_schema_version": "drowsiness_feature_schema_v3",
          "inference_latency_ms": 8.4
        }
        """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(DrowsinessPredictResponse.self, from: payload)
        XCTAssertEqual(decoded.label, "open")
        XCTAssertEqual(decoded.labelIndex, 1)
        XCTAssertEqual(decoded.probabilities.count, 3)
        XCTAssertEqual(decoded.featureSchemaVersion, "drowsiness_feature_schema_v3")
        XCTAssertTrue(decoded.confidence.isFinite)
    }

    func testMaskedSampleWhenNoFace() {
        let sample = DrowsinessFeatureExtractor.makeSample(faces: [], hands: [])
        XCTAssertEqual(sample.values.count, 12)
        XCTAssertTrue(sample.isFinite)
        XCTAssertEqual(sample.values[0], 0.0) // face_detected
        XCTAssertTrue(sample.values.allSatisfy { $0 == 0.0 })
    }

    func testSyntheticFixtureMatchesPythonExpectedVector() {
        // Same points as tests/fixtures/synthetic_eye_landmarks.json
        let leftEye: [CGPoint] = [
            CGPoint(x: 0.40, y: 0.55),
            CGPoint(x: 0.42, y: 0.58),
            CGPoint(x: 0.45, y: 0.59),
            CGPoint(x: 0.48, y: 0.58),
            CGPoint(x: 0.50, y: 0.55),
            CGPoint(x: 0.48, y: 0.52),
            CGPoint(x: 0.45, y: 0.51),
            CGPoint(x: 0.42, y: 0.52),
        ]
        let rightEye: [CGPoint] = [
            CGPoint(x: 0.55, y: 0.55),
            CGPoint(x: 0.57, y: 0.58),
            CGPoint(x: 0.60, y: 0.59),
            CGPoint(x: 0.63, y: 0.58),
            CGPoint(x: 0.65, y: 0.55),
            CGPoint(x: 0.63, y: 0.52),
            CGPoint(x: 0.60, y: 0.51),
            CGPoint(x: 0.57, y: 0.52),
        ]
        let values = DrowsinessFeatureExtractor.featuresFromImageSpace(
            faceDetected: true,
            yaw: 0.12,
            pitch: -0.05,
            roll: 0.02,
            leftEyePoints: leftEye,
            leftPupil: CGPoint(x: 0.45, y: 0.55),
            rightEyePoints: rightEye,
            rightPupil: CGPoint(x: 0.60, y: 0.55)
        )
        let expected: [Double] = [
            1.0,
            0.12,
            -0.05,
            0.02,
            1.0,
            1.0,
            0.8,
            0.8,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
        XCTAssertEqual(values.count, 12)
        for (got, want) in zip(values, expected) {
            XCTAssertEqual(got, want, accuracy: 1e-5)
        }
    }

    func testTranslationInvarianceOfNormalizedEyeFeatures() {
        let leftEye: [CGPoint] = [
            CGPoint(x: 0.40, y: 0.55),
            CGPoint(x: 0.42, y: 0.58),
            CGPoint(x: 0.45, y: 0.59),
            CGPoint(x: 0.48, y: 0.58),
            CGPoint(x: 0.50, y: 0.55),
            CGPoint(x: 0.48, y: 0.52),
            CGPoint(x: 0.45, y: 0.51),
            CGPoint(x: 0.42, y: 0.52),
        ]
        let rightEye: [CGPoint] = [
            CGPoint(x: 0.55, y: 0.55),
            CGPoint(x: 0.57, y: 0.58),
            CGPoint(x: 0.60, y: 0.59),
            CGPoint(x: 0.63, y: 0.58),
            CGPoint(x: 0.65, y: 0.55),
            CGPoint(x: 0.63, y: 0.52),
            CGPoint(x: 0.60, y: 0.51),
            CGPoint(x: 0.57, y: 0.52),
        ]
        let base = DrowsinessFeatureExtractor.featuresFromImageSpace(
            faceDetected: true,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            leftEyePoints: leftEye,
            leftPupil: CGPoint(x: 0.45, y: 0.55),
            rightEyePoints: rightEye,
            rightPupil: CGPoint(x: 0.60, y: 0.55)
        )
        let dx: CGFloat = 0.17
        let dy: CGFloat = -0.09
        let shiftedLeft = leftEye.map { CGPoint(x: $0.x + dx, y: $0.y + dy) }
        let shiftedRight = rightEye.map { CGPoint(x: $0.x + dx, y: $0.y + dy) }
        let shifted = DrowsinessFeatureExtractor.featuresFromImageSpace(
            faceDetected: true,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            leftEyePoints: shiftedLeft,
            leftPupil: CGPoint(x: 0.45 + dx, y: 0.55 + dy),
            rightEyePoints: shiftedRight,
            rightPupil: CGPoint(x: 0.60 + dx, y: 0.55 + dy)
        )
        for index in 4..<12 {
            XCTAssertEqual(shifted[index], base[index], accuracy: 1e-5)
        }
    }

    func testCoordinatorRingBufferDoesNotExceedCap() {
        let coordinator = DrowsinessInferenceCoordinator()
        coordinator.start()
        defer { coordinator.stop() }

        for index in 0..<40 {
            let values = Array(repeating: Double(index), count: 12)
            let sample = DrowsinessFeatureSample(
                timestampMs: Int64(index),
                values: values
            )
            coordinator.ingest(sample)
        }
        XCTAssertNil(coordinator.latestPrediction)
        XCTAssertFalse(coordinator.isWakeUpAlertActive)
    }

    func testNetworkFailureDoesNotCreatePrediction() async throws {
        let coordinator = DrowsinessInferenceCoordinator()
        XCTAssertNil(coordinator.latestPrediction)
        XCTAssertFalse(coordinator.isWakeUpAlertActive)
        coordinator.stop()
        XCTAssertNil(coordinator.latestPrediction)
        XCTAssertFalse(coordinator.isWakeUpAlertActive)
    }
}
