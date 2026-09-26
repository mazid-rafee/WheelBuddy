//
//  DrowsinessFeatureSample.swift
//  WheelBuddy
//

import Foundation

/// One Apple Vision–derived feature row matching the Python training CSV schema.
struct DrowsinessFeatureSample: Equatable, Sendable {
    /// Wall-clock milliseconds since epoch (UTC).
    let timestampMs: Int64
    /// Values in `DrowsinessFeatureContract.featureNames` order.
    let values: [Double]

    var isFinite: Bool {
        values.count == DrowsinessFeatureContract.featureCount
            && values.allSatisfy { $0.isFinite }
    }
}

/// Exact feature contract from `feature_contract.py` (schema v3).
enum DrowsinessFeatureContract {
    /// Must match Python `FEATURE_SCHEMA_VERSION`.
    static let schemaVersion: String = "drowsiness_feature_schema_v3"

    static let featureNames: [String] = [
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

    static var featureCount: Int { featureNames.count }

    /// Camera target FPS in `MultiCamManager` (not stored in the checkpoint).
    static let samplingRateHz: Double = 15.0

    /// Temporal window from training checkpoint ``window_frames``.
    static let windowFrames: Int = 20

    static let eps: Double = 1e-6
    static let minEyeLandmarkPoints: Int = 4

    /// Canonical server class names (must match Python ``CLASS_TO_IDX``).
    static let classNames: [String] = ["closed", "open", "undefined"]
}
