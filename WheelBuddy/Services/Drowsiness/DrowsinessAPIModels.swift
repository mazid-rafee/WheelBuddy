//
//  DrowsinessAPIModels.swift
//  WheelBuddy
//

import Foundation

// Wire models for the remote drowsiness API. JSON keys are snake_case via explicit `CodingKeys`;
// dates use ISO-8601 (configured in `DrowsinessAPIClient`).

/// One `/predict` request body: a fixed-length window of feature rows plus the schema version,
/// sampling rate, and feature names describing their layout.
struct DrowsinessPredictRequest: Codable, Equatable, Sendable {
    let featureSchemaVersion: String
    /// Client-generated per-run ID; echoed back in the response.
    let sessionID: String
    /// Per-session request counter; echoed back so the client can discard stale replies.
    let sequenceID: Int
    let sentAtUTC: Date
    let samplingRateHz: Double
    /// Column names for `DrowsinessAPISample.values`, in order.
    let featureNames: [String]
    let samples: [DrowsinessAPISample]

    enum CodingKeys: String, CodingKey {
        case featureSchemaVersion = "feature_schema_version"
        case sessionID = "session_id"
        case sequenceID = "sequence_id"
        case sentAtUTC = "sent_at_utc"
        case samplingRateHz = "sampling_rate_hz"
        case featureNames = "feature_names"
        case samples
    }
}

/// A single feature row on the wire.
struct DrowsinessAPISample: Codable, Equatable, Sendable {
    /// Unix epoch milliseconds.
    let timestampMs: Int64
    let values: [Double]

    enum CodingKeys: String, CodingKey {
        case timestampMs = "timestamp_ms"
        case values
    }
}

/// Successful `/predict` reply for one window.
struct DrowsinessPredictResponse: Codable, Equatable, Sendable {
    let sessionID: String
    let sequenceID: Int
    /// Predicted class name (e.g. `"closed"`, which drives the wake-up alert).
    let label: String
    let labelIndex: Int
    let confidence: Double
    /// Per-class probabilities keyed by label name.
    let probabilities: [String: Double]
    let modelVersion: String
    /// Optional so replies from servers that omit it still decode.
    let featureSchemaVersion: String?
    /// Server-reported inference latency in milliseconds (network round trip is measured separately).
    let inferenceLatencyMs: Double

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case sequenceID = "sequence_id"
        case label
        case labelIndex = "label_index"
        case confidence
        case probabilities
        case modelVersion = "model_version"
        case featureSchemaVersion = "feature_schema_version"
        case inferenceLatencyMs = "inference_latency_ms"
    }
}

/// Structured error body (`{"error": {"code", "message"}}`) returned with non-2xx statuses.
struct DrowsinessAPIErrorEnvelope: Codable, Sendable {
    let error: DrowsinessAPIErrorBody
}

struct DrowsinessAPIErrorBody: Codable, Sendable {
    let code: String
    let message: String
}
