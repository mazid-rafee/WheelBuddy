//
//  NavigationMode.swift
//  WheelBuddy
//

import Foundation

/// Conceptual modes: CHOOSE LOCATION → SHOW ROUTE → NAVIGATION.
enum NavigationMode: Equatable {
    case chooseLocation
    case showRoute
    case navigation
}

enum RouteLoadingState: Equatable {
    case idle
    case loading
    case ready
    case failed(String)

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }

    var isLoading: Bool {
        if case .loading = self { return true }
        return false
    }

    var failureMessage: String? {
        if case .failed(let message) = self { return message }
        return nil
    }
}

/// One flattened navigation step retained with a preview route candidate.
struct ComputedRouteStep: Equatable, Sendable {
    var encodedPolyline: String
    var distanceMeters: Int
    var staticDurationSeconds: Double
}

struct ComputedRoute: Equatable, Identifiable, Sendable {
    /// Deterministic Google response-order ID (`route_0`, `route_1`, …). Also used as `Identifiable.id`.
    var id: String
    var encodedPolyline: String
    var durationText: String
    /// Traffic-aware overall duration in seconds when parsing succeeded.
    var durationSeconds: Double?
    var distanceMeters: Int
    var sourcePlaceID: String
    var destinationPlaceID: String
    var routeLabels: [String]
    var isDefault: Bool
    var responseIndex: Int
    /// Ordered steps flattened across all legs.
    var steps: [ComputedRouteStep]
    /// Whether this candidate can produce a backend extraction payload.
    var isExtractionReady: Bool
    /// Route risk (sum of adjusted per-cell severity). Lower = safer. Display as `1 − safetyScore`.
    /// `0` when safety info is insufficient. `nil` until scored.
    var safetyScore: Double?
    /// Relative safety tier among informative routes. `nil` when insufficient / unscored (gray UI).
    var safetyTier: RouteSafetyTier?
    /// True when OOV H3 cells exceed 97% of the route (not enough model coverage).
    var hasInsufficientSafetyInfo: Bool

    var distanceMilesText: String {
        let miles = Double(distanceMeters) / 1609.344
        return String(format: "%.1f mi", miles)
    }

    var summaryText: String {
        "\(durationText) · \(distanceMilesText)"
    }
}
