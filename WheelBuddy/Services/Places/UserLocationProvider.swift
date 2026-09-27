//
//  UserLocationProvider.swift
//  WheelBuddy
//

import Combine
import CoreLocation
import Foundation

/// Shared Core Location access for Places bias and “Your location”.
@MainActor
final class UserLocationProvider: NSObject, ObservableObject {
    /// Most recent one-shot fix (or cached `manager.location`); retained across failed updates.
    @Published private(set) var coordinate: CLLocationCoordinate2D?
    @Published private(set) var authorizationStatus: CLAuthorizationStatus
    /// User-facing hint set when location access is denied or restricted.
    @Published private(set) var locationDeniedMessage: String?

    private let manager = CLLocationManager()

    override init() {
        authorizationStatus = manager.authorizationStatus
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBest
        refreshCachedLocation()
    }

    var hasValidCoordinate: Bool {
        guard let coordinate else { return false }
        return CLLocationCoordinate2DIsValid(coordinate)
    }

    func clearDeniedMessage() {
        locationDeniedMessage = nil
    }

    /// Prompts for When-In-Use permission if undetermined, otherwise requests a single location
    /// fix (`requestLocation`, not continuous updates). Sets `locationDeniedMessage` when blocked.
    func requestWhenInUseIfNeeded() {
        locationDeniedMessage = nil
        switch manager.authorizationStatus {
        case .notDetermined:
            manager.requestWhenInUseAuthorization()
        case .authorizedAlways, .authorizedWhenInUse:
            manager.requestLocation()
        case .denied, .restricted:
            locationDeniedMessage = "Location access is off. Search for a place instead."
        @unknown default:
            break
        }
    }

    /// Adopts Core Location's cached fix without a new request; leaves `coordinate` untouched if none.
    func refreshCachedLocation() {
        if let location = manager.location, CLLocationCoordinate2DIsValid(location.coordinate) {
            coordinate = location.coordinate
        }
    }

    private func beginUpdatesIfAuthorized() {
        switch manager.authorizationStatus {
        case .authorizedAlways, .authorizedWhenInUse:
            locationDeniedMessage = nil
            manager.requestLocation()
        case .denied, .restricted:
            locationDeniedMessage = "Location access is off. Search for a place instead."
        default:
            break
        }
    }
}

// Delegate callbacks are nonisolated and hop to the main actor before touching published state.
extension UserLocationProvider: CLLocationManagerDelegate {
    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            self.authorizationStatus = manager.authorizationStatus
            self.beginUpdatesIfAuthorized()
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        Task { @MainActor in
            self.coordinate = location.coordinate
            self.locationDeniedMessage = nil
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        // Keep last known coordinate if available.
    }
}
