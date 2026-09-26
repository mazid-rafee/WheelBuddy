//
//  DrowsinessFeatureExtractor.swift
//  WheelBuddy
//

import Foundation
import Vision

/// Builds schema-v3 training-compatible feature rows from Apple Vision face landmarks.
///
/// Formulas must stay identical to Python `feature_math.py`
/// (`drowsiness_feature_schema_v3`). Eyelid-gap ratios are not part of the model input.
enum DrowsinessFeatureExtractor {
    static func makeSample(
        faces: [VNFaceObservation],
        hands: [VNHumanHandPoseObservation] = [],
        timestamp: Date = Date()
    ) -> DrowsinessFeatureSample {
        // Hands are ignored in v3 model input; parameter retained for call-site stability.
        _ = hands
        let timestampMs = Int64((timestamp.timeIntervalSince1970 * 1000.0).rounded())
        var values = Array(repeating: 0.0, count: DrowsinessFeatureContract.featureCount)

        let face = faces.max(by: {
            ($0.boundingBox.width * $0.boundingBox.height)
                < ($1.boundingBox.width * $1.boundingBox.height)
        })

        guard let face else {
            return DrowsinessFeatureSample(timestampMs: timestampMs, values: values)
        }

        values[0] = 1.0 // face_detected
        values[1] = face.yaw?.doubleValue ?? 0.0
        values[2] = face.pitch?.doubleValue ?? 0.0
        values[3] = face.roll?.doubleValue ?? 0.0

        let landmarks = face.landmarks
        let box = face.boundingBox

        let left = eyeLocalFeatures(
            region: landmarks?.leftEye,
            pupil: landmarks?.leftPupil,
            faceBox: box
        )
        values[4] = left.valid
        values[6] = left.aspectRatio
        values[8] = left.pupilRelX
        values[9] = left.pupilRelY

        let right = eyeLocalFeatures(
            region: landmarks?.rightEye,
            pupil: landmarks?.rightPupil,
            faceBox: box
        )
        values[5] = right.valid
        values[7] = right.aspectRatio
        values[10] = right.pupilRelX
        values[11] = right.pupilRelY

        for index in values.indices where !values[index].isFinite {
            values[index] = 0.0
        }


        return DrowsinessFeatureSample(timestampMs: timestampMs, values: values)
    }

    /// Pure math entry point for fixture parity tests (image-space points).
    static func featuresFromImageSpace(
        faceDetected: Bool,
        yaw: Double,
        pitch: Double,
        roll: Double,
        leftEyePoints: [CGPoint],
        leftPupil: CGPoint?,
        rightEyePoints: [CGPoint],
        rightPupil: CGPoint?
    ) -> [Double] {
        guard faceDetected else {
            return Array(repeating: 0.0, count: DrowsinessFeatureContract.featureCount)
        }
        let left = eyeLocalFeatures(points: leftEyePoints, pupil: leftPupil)
        let right = eyeLocalFeatures(points: rightEyePoints, pupil: rightPupil)
        var values = Array(repeating: 0.0, count: DrowsinessFeatureContract.featureCount)
        values[0] = 1.0
        values[1] = yaw.isFinite ? yaw : 0.0
        values[2] = pitch.isFinite ? pitch : 0.0
        values[3] = roll.isFinite ? roll : 0.0
        values[4] = left.valid
        values[5] = right.valid
        values[6] = left.aspectRatio
        values[7] = right.aspectRatio
        values[8] = left.pupilRelX
        values[9] = left.pupilRelY
        values[10] = right.pupilRelX
        values[11] = right.pupilRelY
        for index in values.indices where !values[index].isFinite {
            values[index] = 0.0
        }
        return values
    }

    // MARK: - Geometry

    private struct EyeLocalFeatures {
        var valid: Double
        var aspectRatio: Double
        var pupilRelX: Double
        var pupilRelY: Double

        static let invalid = EyeLocalFeatures(
            valid: 0,
            aspectRatio: 0,
            pupilRelX: 0,
            pupilRelY: 0
        )
    }

    private static func eyeLocalFeatures(
        region: VNFaceLandmarkRegion2D?,
        pupil: VNFaceLandmarkRegion2D?,
        faceBox: CGRect
    ) -> EyeLocalFeatures {
        guard let region, region.pointCount >= DrowsinessFeatureContract.minEyeLandmarkPoints else {
            return .invalid
        }
        let points = region.normalizedPoints.map { imagePoint($0, faceBox: faceBox) }
        var pupilPoint: CGPoint?
        if let pupil, pupil.pointCount >= 1 {
            pupilPoint = imagePoint(pupil.normalizedPoints[0], faceBox: faceBox)
        }
        return eyeLocalFeatures(points: points, pupil: pupilPoint)
    }

    private static func eyeLocalFeatures(
        points: [CGPoint],
        pupil: CGPoint?
    ) -> EyeLocalFeatures {
        guard points.count >= DrowsinessFeatureContract.minEyeLandmarkPoints else {
            return .invalid
        }
        guard points.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else {
            return .invalid
        }
        guard let xsMin = points.map(\.x).min(),
              let xsMax = points.map(\.x).max(),
              let ysMin = points.map(\.y).min(),
              let ysMax = points.map(\.y).max() else {
            return .invalid
        }
        let width = Double(xsMax - xsMin)
        let height = Double(ysMax - ysMin)
        let eps = DrowsinessFeatureContract.eps
        guard width > eps, height > eps else {
            return .invalid
        }

        let ear = height / max(width, eps)
        guard ear.isFinite else {
            return .invalid
        }

        guard let pupil, pupil.x.isFinite, pupil.y.isFinite else {
            return .invalid
        }
        var relX = (Double(pupil.x) - Double(xsMin)) / max(width, eps)
        var relY = (Double(pupil.y) - Double(ysMin)) / max(height, eps)
        guard relX.isFinite, relY.isFinite else {
            return .invalid
        }
        relX = min(1.0, max(0.0, relX))
        relY = min(1.0, max(0.0, relY))

        return EyeLocalFeatures(
            valid: 1.0,
            aspectRatio: ear,
            pupilRelX: relX,
            pupilRelY: relY
        )
    }

    /// Landmark points are relative to the face bounding box; convert to image coords.
    private static func imagePoint(_ normalizedInFace: CGPoint, faceBox: CGRect) -> CGPoint {
        CGPoint(
            x: faceBox.origin.x + normalizedInFace.x * faceBox.width,
            y: faceBox.origin.y + normalizedInFace.y * faceBox.height
        )
    }
}
