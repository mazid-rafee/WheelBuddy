//
//  GoogleRoutesModels.swift
//  WheelBuddy
//

import Foundation

struct RoutesComputeRequest: Encodable {
    struct LatLng: Encodable {
        var latitude: Double
        var longitude: Double
    }

    struct Location: Encodable {
        var latLng: LatLng
    }

    struct Waypoint: Encodable {
        var location: Location
    }

    var origin: Waypoint
    var destination: Waypoint
    var travelMode: String
    var routingPreference: String
    var computeAlternativeRoutes: Bool
    var polylineQuality: String
    var languageCode: String
    var units: String
}

struct RoutesComputeResponse: Decodable {
    struct Polyline: Decodable {
        var encodedPolyline: String?
    }

    struct Step: Decodable {
        var distanceMeters: Int?
        var staticDuration: String?
        var polyline: Polyline?
    }

    struct Leg: Decodable {
        var distanceMeters: Int?
        var duration: String?
        var steps: [Step]?
    }

    struct Route: Decodable {
        var duration: String?
        var distanceMeters: Int?
        var polyline: Polyline?
        var routeLabels: [String]?
        var legs: [Leg]?
    }

    var routes: [Route]?
}

struct RoutesAPIErrorResponse: Decodable {
    struct Status: Decodable {
        var code: Int?
        var message: String?
        var status: String?
    }

    var error: Status?
}

/// One ordered step taken from Google Routes legs (already flattened).
struct RouteStepCandidate: Equatable, Sendable {
    var encodedPolyline: String
    var distanceMeters: Int
    var staticDurationSeconds: Double
}

/// Intermediate Routes API candidate before session attaches place IDs.
struct RouteCandidateResult: Equatable, Sendable {
    /// Deterministic ID matching Google response order (`route_0`, `route_1`, …).
    var routeID: String
    var encodedPolyline: String
    var durationText: String
    /// Traffic-aware overall duration in seconds; `nil` when the API string is malformed.
    var durationSeconds: Double?
    var distanceMeters: Int
    var routeLabels: [String]
    var isDefault: Bool
    var responseIndex: Int
    /// Flattened steps across all legs, preserving leg then step order.
    var steps: [RouteStepCandidate]
    /// `false` when step / duration extraction is incomplete; map display still uses this candidate.
    var isExtractionReady: Bool
}

/// Result of one Compute Routes request, including the shared client departure timestamp.
struct RoutesComputeBatchResult: Equatable, Sendable {
    var departureTime: Date
    var candidates: [RouteCandidateResult]
}
