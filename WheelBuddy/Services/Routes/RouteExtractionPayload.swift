//
//  RouteExtractionPayload.swift
//  WheelBuddy
//

import Foundation

/// Backend-ready route timing payload derived from a Google Routes candidate.
struct RouteExtractionPayload: Codable, Equatable, Sendable {
    let routeID: String
    /// ISO-8601 UTC string, shared by every route from the same Compute Routes request.
    let departureTimeUTC: String
    /// Traffic-aware total route duration.
    let durationSeconds: Double
    let steps: [RouteStepExtractionPayload]

    enum CodingKeys: String, CodingKey {
        case routeID = "route_id"
        case departureTimeUTC = "departure_time_utc"
        case durationSeconds = "duration_seconds"
        case steps
    }
}

/// One flattened route step; `staticDurationSeconds` is Google's traffic-free step duration.
struct RouteStepExtractionPayload: Codable, Equatable, Sendable {
    let encodedPolyline: String
    let distanceMeters: Int
    let staticDurationSeconds: Double

    enum CodingKeys: String, CodingKey {
        case encodedPolyline = "encoded_polyline"
        case distanceMeters = "distance_meters"
        case staticDurationSeconds = "static_duration_seconds"
    }
}

/// Builds backend extraction payloads from loaded route candidates and a shared departure time.
enum RouteExtractionBuilder {
    /// Converts route candidates into backend payloads, preserving Google response order.
    /// Routes with malformed extraction metadata are skipped (map display is unaffected).
    static func buildPayloads(
        routes: [ComputedRoute],
        departureTime: Date
    ) -> [RouteExtractionPayload] {
        let departureTimeUTC = RouteDepartureTimeFormatting.iso8601UTCString(from: departureTime)
        return routes
            .sorted { $0.responseIndex < $1.responseIndex }
            .compactMap { route in
                guard route.isExtractionReady,
                      let durationSeconds = route.durationSeconds else {
                    #if DEBUG
                    print(
                        "[ROUTE_EXTRACTION] skipping malformed extraction for \(route.id) (responseIndex=\(route.responseIndex))"
                    )
                    #endif
                    return nil
                }

                return RouteExtractionPayload(
                    routeID: route.id,
                    departureTimeUTC: departureTimeUTC,
                    durationSeconds: durationSeconds,
                    steps: route.steps.map {
                        RouteStepExtractionPayload(
                            encodedPolyline: $0.encodedPolyline,
                            distanceMeters: $0.distanceMeters,
                            staticDurationSeconds: $0.staticDurationSeconds
                        )
                    }
                )
            }
    }

    #if DEBUG
    static func debugJSONString(for payloads: [RouteExtractionPayload]) -> String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(payloads),
              let json = String(data: data, encoding: .utf8) else {
            return nil
        }
        return json
    }

    static func logPayloads(_ payloads: [RouteExtractionPayload]) {
        guard let json = debugJSONString(for: payloads) else {
            print("[ROUTE_EXTRACTION] failed to encode payloads")
            return
        }
        print("[ROUTE_EXTRACTION]\n\(json)")
    }
    #endif
}
