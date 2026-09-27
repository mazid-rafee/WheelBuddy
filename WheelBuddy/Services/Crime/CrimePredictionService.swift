//
//  CrimePredictionService.swift
//  WheelBuddy
//

import Foundation

// MARK: - Errors

/// Failures from `CrimePredictionService`, covering transport, server-reported errors, and
/// client-side validation of the response contract.
enum CrimePredictionServiceError: LocalizedError, Sendable {
    case invalidBaseURL
    case invalidResponse
    case httpStatus(Int, String)
    case apiError(code: String, message: String, details: CrimeAPIErrorDetails?)
    case requestIDMismatch(expected: String, actual: String)
    case unexpectedRouteIDs(String)
    case duplicateRouteIDs
    case nonFinitePrediction(String)
    case decoding(Error)

    var errorDescription: String? {
        switch self {
        case .invalidBaseURL:
            return "Crime API base URL is invalid."
        case .invalidResponse:
            return "Crime API returned an invalid response."
        case let .httpStatus(code, body):
            return "Crime API HTTP \(code): \(body)"
        case let .apiError(code, message, _):
            return "Crime API error (\(code)): \(message)"
        case let .requestIDMismatch(expected, actual):
            return "Crime API request_id mismatch (expected \(expected), got \(actual))."
        case let .unexpectedRouteIDs(detail):
            return "Crime API returned unexpected route IDs: \(detail)"
        case .duplicateRouteIDs:
            return "Crime API returned duplicate route IDs."
        case let .nonFinitePrediction(detail):
            return "Crime API returned a non-finite prediction (\(detail))."
        case let .decoding(error):
            return "Crime API decoding failed: \(error.localizedDescription)"
        }
    }

    /// Structured server details (e.g. per-route OOV stats), only present for `.apiError`.
    var apiErrorDetails: CrimeAPIErrorDetails? {
        if case let .apiError(_, _, details) = self {
            return details
        }
        return nil
    }
}

// MARK: - Service

/// Capture-only client for the local CrimePredictor FastAPI server.
actor CrimePredictionService {
    static let shared = CrimePredictionService()

    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    /// Pass a custom `session` for tests; otherwise uses an ephemeral session (60 s request /
    /// 120 s resource timeouts) that fails immediately instead of waiting for connectivity.
    init(session: URLSession? = nil) {
        if let session {
            self.session = session
        } else {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.timeoutIntervalForRequest = 60
            configuration.timeoutIntervalForResource = 120
            configuration.waitsForConnectivity = false
            self.session = URLSession(configuration: configuration)
        }

        decoder = JSONDecoder()
        encoder = JSONEncoder()
        // RouteExtractionPayload already uses explicit snake_case CodingKeys.
    }

    // MARK: Public API

    /// POSTs all routes in one request and validates the structured response.
    /// - Throws: `CancellationError` if the task (or underlying URL request) is cancelled;
    ///   `CrimePredictionServiceError` for non-2xx responses (`.apiError` when the FastAPI error
    ///   envelope decodes, otherwise `.httpStatus`), decoding failures, or validation failures.
    ///   Other transport errors propagate unchanged.
    func predictRoutes(_ request: RoutePredictionRequest) async throws -> RoutePredictionResponse {
        try Task.checkCancellation()

        let url = CrimeAPIConfiguration.predictRoutesURL
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.timeoutInterval = 60
        urlRequest.httpBody = try encoder.encode(request)

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: urlRequest)
        } catch is CancellationError {
            throw CancellationError()
        } catch let urlError as URLError where urlError.code == .cancelled {
            throw CancellationError()
        }

        try Task.checkCancellation()

        guard let http = response as? HTTPURLResponse else {
            throw CrimePredictionServiceError.invalidResponse
        }

        guard (200..<300).contains(http.statusCode) else {
            if let envelope = try? decoder.decode(CrimeAPIErrorEnvelope.self, from: data) {
                throw CrimePredictionServiceError.apiError(
                    code: envelope.error.code,
                    message: envelope.error.message,
                    details: envelope.error.details
                )
            }
            let body = String(data: data, encoding: .utf8) ?? "<non-utf8 body>"
            throw CrimePredictionServiceError.httpStatus(http.statusCode, body)
        }

        let decoded: RoutePredictionResponse
        do {
            decoded = try decoder.decode(RoutePredictionResponse.self, from: data)
        } catch {
            throw CrimePredictionServiceError.decoding(error)
        }

        try validate(response: decoded, against: request)
        return decoded
    }

    // MARK: Validation

    /// Enforces the response contract: matching `request_id`, exactly the requested route IDs
    /// (no duplicates, none missing or extra), and finite summary, time-bin, and per-cell rates.
    private func validate(
        response: RoutePredictionResponse,
        against request: RoutePredictionRequest
    ) throws {
        guard response.requestID == request.requestID else {
            throw CrimePredictionServiceError.requestIDMismatch(
                expected: request.requestID,
                actual: response.requestID
            )
        }

        let expectedIDs = Set(request.routes.map(\.routeID))
        let returnedIDs = response.routes.map(\.routeID)
        if Set(returnedIDs).count != returnedIDs.count {
            throw CrimePredictionServiceError.duplicateRouteIDs
        }
        let returnedSet = Set(returnedIDs)
        guard returnedSet == expectedIDs else {
            let missing = expectedIDs.subtracting(returnedSet)
            let unexpected = returnedSet.subtracting(expectedIDs)
            throw CrimePredictionServiceError.unexpectedRouteIDs(
                "missing=\(Array(missing).sorted()) unexpected=\(Array(unexpected).sorted())"
            )
        }

        for route in response.routes {
            try assertFinite(route.predictionSummary.mean, label: "\(route.routeID).mean")
            try assertFinite(route.predictionSummary.maximum, label: "\(route.routeID).maximum")
            try assertFinite(route.predictionSummary.sum, label: "\(route.routeID).sum")
            for binScore in route.timeBinScores {
                let prefix = "\(route.routeID).bin\(binScore.hourBinStart)"
                try assertFinite(binScore.severityWeightedSum, label: "\(prefix).severity_weighted_sum")
                if let adjusted = binScore.adjustedSeverityWeightedSum {
                    try assertFinite(adjusted, label: "\(prefix).adjusted_severity_weighted_sum")
                }
                try assertFinite(binScore.maxSeverityWeightedRate, label: "\(prefix).max_severity_weighted_rate")
                try assertFinite(binScore.meanPersonRate, label: "\(prefix).mean_person_rate")
                try assertFinite(binScore.meanPropertyRate, label: "\(prefix).mean_property_rate")
                try assertFinite(binScore.meanSocietyRate, label: "\(prefix).mean_society_rate")
                try assertFinite(binScore.meanOtherRate, label: "\(prefix).mean_other_rate")
            }
            for cell in route.cells {
                try assertFinite(cell.severityWeightedRate, label: "\(route.routeID).severity_weighted_rate")
                if let highHour = cell.highHourSeverityWeightedRate {
                    try assertFinite(highHour, label: "\(route.routeID).high_hour_severity_weighted_rate")
                }
                try assertFinite(cell.totalRate, label: "\(route.routeID).total_rate")
                try assertFinite(cell.personRate, label: "\(route.routeID).person_rate")
                try assertFinite(cell.propertyRate, label: "\(route.routeID).property_rate")
                try assertFinite(cell.societyRate, label: "\(route.routeID).society_rate")
                try assertFinite(cell.otherRate, label: "\(route.routeID).other_rate")
            }
        }
    }

    private func assertFinite(_ value: Double, label: String) throws {
        guard value.isFinite else {
            throw CrimePredictionServiceError.nonFinitePrediction(label)
        }
    }
}
