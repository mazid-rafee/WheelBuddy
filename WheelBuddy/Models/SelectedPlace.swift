//
//  SelectedPlace.swift
//  WheelBuddy
//

import CoreLocation
import Foundation

/// Reusable place selection used by source / destination search.
struct SelectedPlace: Equatable, Identifiable {
    /// Stable ID for Places results; `"current_location"` for the device location option.
    var placeID: String
    var primaryDisplayName: String
    var formattedAddress: String
    var coordinate: CLLocationCoordinate2D

    var id: String { placeID }

    var isCurrentLocation: Bool {
        placeID == Self.currentLocationPlaceID
    }

    static let currentLocationPlaceID = "current_location"

    static func currentLocation(coordinate: CLLocationCoordinate2D) -> SelectedPlace {
        SelectedPlace(
            placeID: currentLocationPlaceID,
            primaryDisplayName: "Your location",
            formattedAddress: "Current location",
            coordinate: coordinate
        )
    }

    static func == (lhs: SelectedPlace, rhs: SelectedPlace) -> Bool {
        lhs.placeID == rhs.placeID
            && lhs.primaryDisplayName == rhs.primaryDisplayName
            && lhs.formattedAddress == rhs.formattedAddress
            && lhs.coordinate.latitude == rhs.coordinate.latitude
            && lhs.coordinate.longitude == rhs.coordinate.longitude
    }
}

/// Lightweight autocomplete row model (independent of GMS types for SwiftUI).
struct PlacePredictionItem: Identifiable, Equatable {
    var id: String { placeID }
    var placeID: String
    var primaryText: String
    var secondaryText: String
}
