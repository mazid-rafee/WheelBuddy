//
//  DrowsinessAPIModels.swift
//  WheelBuddy
//

import Foundation

struct DrowsinessPredictRequest: Codable, Equatable, Sendable {
    let featureSchemaVersion: String
    let sessionID: String
    let sequenceID: Int
    let sentAtUTC: Date
    let samplingRateHz: Double
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

struct DrowsinessAPISample: Codable, Equatable, Sendable {
    let timestampMs: Int64
    let values: [Double]

    enum CodingKeys: String, CodingKey {
        case timestampMs = "timestamp_ms"
        case values
    }
}

struct DrowsinessPredictResponse: Codable, Equatable, Sendable {
    let sessionID: String
    let sequenceID: Int
    let label: String
    let labelIndex: Int
    let confidence: Double
    let probabilities: [String: Double]
    let modelVersion: String
    let featureSchemaVersion: String?
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

struct DrowsinessAPIErrorEnvelope: Codable, Sendable {
    let error: DrowsinessAPIErrorBody
}

struct DrowsinessAPIErrorBody: Codable, Sendable {
    let code: String
    let message: String
}
