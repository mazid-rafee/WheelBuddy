//
//  GoogleRoutesService.swift
//  WheelBuddy
//

import CoreLocation
import Foundation

// MARK: - Errors

/// Failures surfaced by `GoogleRoutesService`; `errorDescription` is user-presentable.
enum GoogleRoutesError: LocalizedError {
    case missingAPIKey
    case invalidCoordinates
    case emptyRoutes
    case httpStatus(Int, String)
    case decoding(String)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "Routes API key is not configured."
        case .invalidCoordinates:
            return "Invalid start or destination coordinates."
        case .emptyRoutes:
            return "No driving route was returned."
        case .httpStatus(_, let message):
            return message
        case .decoding(let message):
            return message
        case .cancelled:
            return "Route request was cancelled."
        }
    }
}

// MARK: - Service

/// Google Routes API v2 Compute Routes client.
actor GoogleRoutesService {
    /// Response field mask sent as `X-Goog-FieldMask`. Only these fields are returned by Google,
    /// so any new field read from `RoutesComputeResponse` must also be added here.
    static let fieldMask = [
        "routes.distanceMeters",
        "routes.duration",
        "routes.polyline.encodedPolyline",
        "routes.routeLabels",
        "routes.legs.distanceMeters",
        "routes.legs.duration",
        "routes.legs.steps.distanceMeters",
        "routes.legs.steps.staticDuration",
        "routes.legs.steps.polyline.encodedPolyline"
    ].joined(separator: ",")

    private let session: URLSession
    private let endpoint = URL(string: "https://routes.googleapis.com/directions/v2:computeRoutes")!

    init(session: URLSession = .shared) {
        self.session = session
    }

    // MARK: Public API

    /// Returns every driving route candidate, with the default route first.
    /// Captures a single departure timestamp immediately before the network request.
    /// - Throws: `GoogleRoutesError` for invalid input, missing key, non-2xx status (using Google's
    ///   error message when parseable), or undecodable / empty responses. Transport errors from
    ///   `URLSession` are propagated unchanged.
    func computeDrivingRoutes(
        from origin: CLLocationCoordinate2D,
        to destination: CLLocationCoordinate2D
    ) async throws -> RoutesComputeBatchResult {
        guard CLLocationCoordinate2DIsValid(origin),
              CLLocationCoordinate2DIsValid(destination) else {
            throw GoogleRoutesError.invalidCoordinates
        }

        let apiKey = RoutesAPIConfiguration.apiKey
        guard !apiKey.isEmpty else {
            throw GoogleRoutesError.missingAPIKey
        }

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "X-Goog-Api-Key")
        request.setValue(Self.fieldMask, forHTTPHeaderField: "X-Goog-FieldMask")
        if let bundleID = Bundle.main.bundleIdentifier {
            request.setValue(bundleID, forHTTPHeaderField: "X-Ios-Bundle-Identifier")
        }

        let body = RoutesComputeRequest(
            origin: .init(location: .init(latLng: .init(
                latitude: origin.latitude,
                longitude: origin.longitude
            ))),
            destination: .init(location: .init(latLng: .init(
                latitude: destination.latitude,
                longitude: destination.longitude
            ))),
            travelMode: "DRIVE",
            routingPreference: "TRAFFIC_AWARE",
            computeAlternativeRoutes: true,
            polylineQuality: "HIGH_QUALITY",
            languageCode: "en-US",
            units: "IMPERIAL"
        )
        request.httpBody = try JSONEncoder().encode(body)

        // One shared departure instant for this request and all returned alternatives.
        let departureTime = Date()
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw GoogleRoutesError.decoding("Invalid Routes API response.")
        }

        guard (200..<300).contains(http.statusCode) else {
            let message = Self.parseErrorMessage(from: data)
                ?? "Routes request failed (\(http.statusCode))."
            throw GoogleRoutesError.httpStatus(http.statusCode, message)
        }

        do {
            let candidates = try Self.decodeCandidates(from: data)
            return RoutesComputeBatchResult(
                departureTime: departureTime,
                candidates: candidates
            )
        } catch let error as GoogleRoutesError {
            throw error
        } catch {
            throw GoogleRoutesError.decoding("Couldn't decode the route response.")
        }
    }

    // MARK: Parsing

    /// Decodes Routes API JSON into map-ready candidates with extraction metadata.
    /// Exposed for unit tests via the same decoding path the live client uses.
    /// Routes lacking a polyline or distance are dropped; routes with incomplete step/duration
    /// data are kept for display but marked `isExtractionReady == false`.
    /// - Throws: `GoogleRoutesError.emptyRoutes` if no usable route remains.
    static func decodeCandidates(from data: Data) throws -> [RouteCandidateResult] {
        let decoded = try JSONDecoder().decode(RoutesComputeResponse.self, from: data)
        let rawRoutes = decoded.routes ?? []
        var candidates: [RouteCandidateResult] = []
        candidates.reserveCapacity(rawRoutes.count)

        for (index, route) in rawRoutes.enumerated() {
            guard let encoded = route.polyline?.encodedPolyline,
                  !encoded.isEmpty,
                  let distance = route.distanceMeters else {
                #if DEBUG
                print("[Routes] skipping route index=\(index) (missing polyline or distance)")
                #endif
                continue
            }

            // IDs use the raw response index, so skipped routes leave gaps (e.g. route_0, route_2).
            let routeID = "route_\(index)"
            let labels = route.routeLabels ?? []
            let durationSeconds = GoogleDurationParser.parseSeconds(route.duration)
            let stepExtraction = flattenSteps(from: route, routeID: routeID)
            let isExtractionReady = durationSeconds != nil
                && stepExtraction.isValid
                && !stepExtraction.steps.isEmpty

            if !isExtractionReady {
                #if DEBUG
                print(
                    "[ROUTE_EXTRACTION] malformed extraction for \(routeID); map route will still be shown"
                )
                #endif
            }

            let candidate = RouteCandidateResult(
                routeID: routeID,
                encodedPolyline: encoded,
                durationText: displayDuration(fromSeconds: durationSeconds, raw: route.duration),
                durationSeconds: durationSeconds,
                distanceMeters: distance,
                routeLabels: labels,
                isDefault: labels.contains("DEFAULT_ROUTE"),
                responseIndex: index,
                steps: stepExtraction.steps,
                isExtractionReady: isExtractionReady
            )
            candidates.append(candidate)

            #if DEBUG
            print(
                "[Routes] index=\(index) id=\(routeID) labels=\(labels) distanceMeters=\(distance) duration=\(candidate.durationText) steps=\(candidate.steps.count) extractionReady=\(isExtractionReady)"
            )
            #endif
        }

        guard !candidates.isEmpty else {
            throw GoogleRoutesError.emptyRoutes
        }

        // If Google omitted labels, treat the first returned route as the default.
        if !candidates.contains(where: \.isDefault) {
            candidates[0].isDefault = true
        }

        // Default route first; alternatives keep Google's response order.
        candidates.sort { lhs, rhs in
            if lhs.isDefault != rhs.isDefault {
                return lhs.isDefault && !rhs.isDefault
            }
            return lhs.responseIndex < rhs.responseIndex
        }

        #if DEBUG
        print("[Routes] received \(candidates.count) valid route candidate(s)")
        #endif

        return candidates
    }

    /// Flattens `legs[].steps[]` in leg order, then step order within each leg.
    /// Stops at the first missing/malformed leg or step and returns `isValid == false` along with
    /// the steps collected so far; callers should not use a partial step list.
    static func flattenSteps(
        from route: RoutesComputeResponse.Route,
        routeID: String
    ) -> (steps: [RouteStepCandidate], isValid: Bool) {
        guard let legs = route.legs else {
            return ([], false)
        }

        var steps: [RouteStepCandidate] = []
        for (legIndex, leg) in legs.enumerated() {
            guard let legSteps = leg.steps else {
                #if DEBUG
                print("[ROUTE_EXTRACTION] \(routeID) leg[\(legIndex)] missing steps")
                #endif
                return (steps, false)
            }

            for (stepIndex, step) in legSteps.enumerated() {
                guard let encoded = step.polyline?.encodedPolyline,
                      !encoded.isEmpty,
                      let distance = step.distanceMeters,
                      let staticDuration = GoogleDurationParser.parseSeconds(step.staticDuration) else {
                    #if DEBUG
                    print(
                        "[ROUTE_EXTRACTION] \(routeID) malformed step at leg[\(legIndex)].step[\(stepIndex)]"
                    )
                    #endif
                    return (steps, false)
                }

                steps.append(
                    RouteStepCandidate(
                        encodedPolyline: encoded,
                        distanceMeters: distance,
                        staticDurationSeconds: staticDuration
                    )
                )
            }
        }

        return (steps, true)
    }

    // MARK: Helpers

    /// Extracts `error.message` from a Google API error body, or `nil` if the body isn't that shape.
    private static func parseErrorMessage(from data: Data) -> String? {
        guard let payload = try? JSONDecoder().decode(RoutesAPIErrorResponse.self, from: data) else {
            return nil
        }
        return payload.error?.message
    }

    /// Converts parsed seconds into a short display value.
    /// Leftover minutes are rounded up; falls back to the raw API string (or "--") when unparsed.
    private static func displayDuration(fromSeconds seconds: Double?, raw: String?) -> String {
        guard let seconds else { return raw ?? "--" }
        let wholeSeconds = Int(seconds.rounded(.towardZero))
        let hours = wholeSeconds / 3600
        let minutes = (wholeSeconds % 3600 + 59) / 60
        if hours > 0 {
            return "\(hours) hr \(minutes) min"
        }
        return "\(max(minutes, 1)) min"
    }
}

// MARK: - Configuration

enum RoutesAPIConfiguration {
    private static let placeholder = "REPLACE_WITH_ROUTES_API_KEY"

    /// Reads the Routes API key from Info.plist (`RoutesAPIKey`, then `ROUTES_API_KEY`), populated via Secrets.xcconfig.
    /// Returns an empty string when unset or still the placeholder value.
    static var apiKey: String {
        let candidates = [
            Bundle.main.object(forInfoDictionaryKey: "RoutesAPIKey") as? String,
            Bundle.main.object(forInfoDictionaryKey: "ROUTES_API_KEY") as? String
        ]
        for raw in candidates {
            let trimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !trimmed.isEmpty, trimmed != placeholder else { continue }
            return trimmed
        }
        return ""
    }
}
