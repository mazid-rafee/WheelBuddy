//
//  RoadsSpeedLimitService.swift
//  WheelBuddy
//
//  Google Roads API speedLimits — posted MPH for the custom speed cluster.
//  Requires Roads API (Speed Limits / Asset Tracking) enabled on the Maps key.
//

import CoreLocation
import Foundation

/// Posted speed limit of the last road segment Google returned for the queried path.
struct RoadsSpeedLimitResult: Equatable, Sendable {
    /// Rounded to the nearest whole MPH; always > 0.
    let speedLimitMPH: Int
    /// Google place ID of the road segment the limit applies to, when provided.
    let placeId: String?
}

enum RoadsSpeedLimitError: LocalizedError {
    case missingAPIKey
    case invalidURL
    case http(Int, String)
    case decoding(String)
    case empty

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "Maps API key missing for Roads speedLimits."
        case .invalidURL:
            return "Invalid Roads speedLimits URL."
        case .http(let code, let body):
            return "Roads speedLimits HTTP \(code): \(body)"
        case .decoding(let detail):
            return "Roads speedLimits decode failed: \(detail)"
        case .empty:
            return "Roads speedLimits returned no limits."
        }
    }
}

/// Thin client for `https://roads.googleapis.com/v1/speedLimits`.
final class RoadsSpeedLimitService: Sendable {
    static let shared = RoadsSpeedLimitService()

    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    /// Asks Google for the posted limit along a short GPS path (snapped server-side).
    /// Only the first 100 points are sent; the limit of the last returned segment is used, so pass
    /// points oldest-first. 12 s request timeout.
    /// - Throws: `RoadsSpeedLimitError` for a missing key, empty path, non-2xx status (body truncated
    ///   to 280 chars), decode failure, or no positive limit. Transport errors propagate unchanged.
    func fetchSpeedLimitMPH(path: [CLLocationCoordinate2D]) async throws -> RoadsSpeedLimitResult {
        let key = RoadsAPIConfiguration.apiKey
        guard !key.isEmpty else { throw RoadsSpeedLimitError.missingAPIKey }
        guard path.count >= 1 else { throw RoadsSpeedLimitError.empty }

        var components = URLComponents(string: "https://roads.googleapis.com/v1/speedLimits")
        let pathValue = path
            .prefix(100)
            .map { String(format: "%.6f,%.6f", $0.latitude, $0.longitude) }
            .joined(separator: "|")
        components?.queryItems = [
            URLQueryItem(name: "path", value: pathValue),
            URLQueryItem(name: "units", value: "MPH"),
            URLQueryItem(name: "key", value: key)
        ]
        guard let url = components?.url else { throw RoadsSpeedLimitError.invalidURL }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 12

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw RoadsSpeedLimitError.http(-1, "non-HTTP response")
        }
        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw RoadsSpeedLimitError.http(http.statusCode, String(body.prefix(280)))
        }

        let payload: SpeedLimitsResponse
        do {
            payload = try JSONDecoder().decode(SpeedLimitsResponse.self, from: data)
        } catch {
            throw RoadsSpeedLimitError.decoding(error.localizedDescription)
        }

        guard let last = payload.speedLimits.last else {
            throw RoadsSpeedLimitError.empty
        }
        let mph = Int(last.speedLimit.rounded())
        guard mph > 0 else { throw RoadsSpeedLimitError.empty }
        return RoadsSpeedLimitResult(speedLimitMPH: mph, placeId: last.placeId)
    }
}

// MARK: - Response models

private struct SpeedLimitsResponse: Decodable, Sendable {
    let speedLimits: [SpeedLimitEntry]
}

private struct SpeedLimitEntry: Decodable, Sendable {
    let placeId: String?
    let speedLimit: Double
    let units: String?
}

// MARK: - Configuration

enum RoadsAPIConfiguration {
    private static let placeholder = "REPLACE_WITH_MAPS_API_KEY"

    /// Prefers the Maps SDK key (`GMSApiKey`); falls back to Routes key.
    /// Skips empty, placeholder, and unexpanded `$(...)` build-setting values; returns "" if none remain.
    static var apiKey: String {
        let candidates = [
            Bundle.main.object(forInfoDictionaryKey: "GMSApiKey") as? String,
            Bundle.main.object(forInfoDictionaryKey: "RoutesAPIKey") as? String,
            Bundle.main.object(forInfoDictionaryKey: "ROUTES_API_KEY") as? String
        ]
        for raw in candidates {
            let trimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !trimmed.isEmpty,
                  trimmed != placeholder,
                  !trimmed.hasPrefix("$(") else { continue }
            return trimmed
        }
        return ""
    }
}
