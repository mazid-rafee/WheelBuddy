//
//  RoutePredictionModels.swift
//  WheelBuddy
//

import Foundation

// MARK: - Request

/// Top-level CrimePredictor `/predict-routes` request wrapping existing extraction payloads.
struct RoutePredictionRequest: Codable, Sendable {
    let requestID: String
    let routes: [RouteExtractionPayload]

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case routes
    }
}

// MARK: - Response

/// `/predict-routes` response; `requestID` and route IDs are validated against the request
/// by `CrimePredictionService`.
struct RoutePredictionResponse: Codable, Sendable, Equatable {
    let requestID: String
    let modelVersion: String
    let routes: [RoutePrediction]

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case modelVersion = "model_version"
        case routes
    }
}

/// Model output for one route, keyed by the `route_id` sent in the request.
struct RoutePrediction: Codable, Sendable, Equatable {
    let routeID: String
    /// Total H3 cells along the densified route (including OOV skips).
    let cellCount: Int
    /// Cells that were scored (in training vocabulary).
    let scoredCellCount: Int
    /// Cells skipped because they were outside the training H3 vocabulary.
    let outOfVocabularyCount: Int
    /// 3-hour bin used for route ranking (from departure local time).
    let activeHourBinStart: Int
    let predictionSummary: RoutePredictionSummary
    /// Safety score for every 3-hour training bin over the same route geometry.
    let timeBinScores: [TimeBinSafetyScore]
    let cells: [CellPrediction]

    enum CodingKeys: String, CodingKey {
        case routeID = "route_id"
        case cellCount = "cell_count"
        case scoredCellCount = "scored_cell_count"
        case outOfVocabularyCount = "out_of_vocabulary_count"
        case activeHourBinStart = "active_hour_bin_start"
        case predictionSummary = "prediction_summary"
        case timeBinScores = "time_bin_scores"
        case cells
    }
}

/// Server-side aggregates of per-cell predictions. `sum` is the fallback route risk used by
/// `RouteRiskScorer` when per-cell P90 values are unavailable.
struct RoutePredictionSummary: Codable, Sendable, Equatable {
    let mean: Double
    let maximum: Double
    let sum: Double
}

/// Route-level safety aggregates for one training 3-hour time bin.
struct TimeBinSafetyScore: Codable, Sendable, Equatable {
    let hourBinStart: Int
    let severityWeightedSum: Double
    /// Unique-cell adjusted route risk for this bin; nil on older CrimePredictor servers.
    let adjustedSeverityWeightedSum: Double?
    let maxSeverityWeightedRate: Double
    let meanPersonRate: Double
    let meanPropertyRate: Double
    let meanSocietyRate: Double
    let meanOtherRate: Double
    let cellCount: Int

    enum CodingKeys: String, CodingKey {
        case hourBinStart = "hour_bin_start"
        case severityWeightedSum = "severity_weighted_sum"
        case adjustedSeverityWeightedSum = "adjusted_severity_weighted_sum"
        case maxSeverityWeightedRate = "max_severity_weighted_rate"
        case meanPersonRate = "mean_person_rate"
        case meanPropertyRate = "mean_property_rate"
        case meanSocietyRate = "mean_society_rate"
        case meanOtherRate = "mean_other_rate"
        case cellCount = "cell_count"
    }
}

/// One H3 cell along a route with model rates (softplus per-hour rates, not probabilities).
struct CellPrediction: Codable, Sendable, Equatable {
    let sequenceIndex: Int
    let h3Cell: String
    let entryTimeUTC: String
    let localHour: Int
    let dayOfWeek: String
    let month: String
    let cityName: String
    let severityWeightedRate: Double
    /// P90 severity across departure weekday/month time bins; nil on older CrimePredictor servers.
    let highHourSeverityWeightedRate: Double?
    let totalRate: Double
    let personRate: Double
    let propertyRate: Double
    let societyRate: Double
    let otherRate: Double
    let hourBinStart: Int

    enum CodingKeys: String, CodingKey {
        case sequenceIndex = "sequence_index"
        case h3Cell = "h3_cell"
        case entryTimeUTC = "entry_time_utc"
        case localHour = "local_hour"
        case dayOfWeek = "day_of_week"
        case month
        case cityName = "city_name"
        case severityWeightedRate = "severity_weighted_rate"
        case highHourSeverityWeightedRate = "high_hour_severity_weighted_rate"
        case totalRate = "total_rate"
        case personRate = "person_rate"
        case propertyRate = "property_rate"
        case societyRate = "society_rate"
        case otherRate = "other_rate"
        case hourBinStart = "hour_bin_start"
    }
}

// MARK: - FastAPI error envelope

/// Error body returned by CrimePredictor on non-2xx responses (`{"error": {...}}`).
struct CrimeAPIErrorEnvelope: Codable, Sendable {
    let error: CrimeAPIErrorBody
}

struct CrimeAPIErrorBody: Codable, Sendable {
    let code: String
    let message: String
    let requestID: String?
    let routeID: String?
    let details: CrimeAPIErrorDetails?

    enum CodingKeys: String, CodingKey {
        case code
        case message
        case requestID = "request_id"
        case routeID = "route_id"
        case details
    }
}

struct CrimeAPIErrorDetails: Codable, Sendable {
    let routes: [CrimeAPIRouteCellStats]?
}

/// Per-route H3 coverage stats attached to errors, useful for diagnosing out-of-vocabulary routes.
struct CrimeAPIRouteCellStats: Codable, Sendable {
    let routeID: String
    let cellCount: Int
    let outOfVocabularyCount: Int
    let exampleOOVCells: [String]?

    enum CodingKeys: String, CodingKey {
        case routeID = "route_id"
        case cellCount = "cell_count"
        case outOfVocabularyCount = "out_of_vocabulary_count"
        case exampleOOVCells = "example_oov_cells"
    }
}

// MARK: - Session state

/// Lifecycle of a crime prediction request; `failed` carries a human-readable error message.
enum RoutePredictionState: Equatable {
    case idle
    case loading
    case ready(RoutePredictionResponse)
    case failed(String)
}

#if DEBUG
enum RoutePredictionDebugLogging {
    static func logSummary(_ response: RoutePredictionResponse) {
        print("[CRIME_PREDICTION] request_id=\(response.requestID)")
        for route in response.routes {
            let summary = route.predictionSummary
            print(
                "[CRIME_PREDICTION] \(route.routeID) cells=\(route.cellCount) "
                    + "scored=\(route.scoredCellCount) oov=\(route.outOfVocabularyCount) "
                    + "active_bin=\(route.activeHourBinStart) "
                    + "mean=\(format(summary.mean)) max=\(format(summary.maximum)) sum=\(format(summary.sum))"
            )
            for binScore in route.timeBinScores {
                let adjusted = binScore.adjustedSeverityWeightedSum.map(format) ?? "nil"
                print(
                    "[CRIME_PREDICTION] \(route.routeID) bin=\(binScore.hourBinStart) "
                        + "sum=\(format(binScore.severityWeightedSum)) adjusted=\(adjusted)"
                )
            }
        }
    }

    static func logFailure(
        requestID: String,
        error: Error
    ) {
        print("[CRIME_PREDICTION] failed request_id=\(requestID) error=\(error.localizedDescription)")
        if let apiError = error as? CrimePredictionServiceError,
           let routes = apiError.apiErrorDetails?.routes {
            for route in routes {
                print(
                    "[CRIME_PREDICTION] \(route.routeID) cells=\(route.cellCount) "
                        + "oov=\(route.outOfVocabularyCount)"
                )
            }
        }
    }

    /// Explicit helper for full per-cell inspection (not used by default).
    static func logAllCells(_ response: RoutePredictionResponse) {
        print("[CRIME_PREDICTION_CELLS] request_id=\(response.requestID)")
        for route in response.routes {
            print("[CRIME_PREDICTION_CELLS] \(route.routeID) cell_count=\(route.cellCount)")
            for cell in route.cells {
                print(
                    "[CRIME_PREDICTION_CELLS] seq=\(cell.sequenceIndex) h3=\(cell.h3Cell) "
                        + "city=\(cell.cityName) hour=\(cell.localHour) bin=\(cell.hourBinStart) "
                        + "severity_weighted_rate=\(format(cell.severityWeightedRate)) "
                        + "total_rate=\(format(cell.totalRate))"
                )
            }
        }
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.6g", value)
    }
}
#endif
