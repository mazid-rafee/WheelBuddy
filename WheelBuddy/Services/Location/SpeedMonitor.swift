//
//  SpeedMonitor.swift
//  WheelBuddy
//

import Combine
import CoreLocation
import Foundation

/// Publishes smoothed GPS vehicle speed (MPH).
/// Uses its own CLLocationManager; does not share Maps / Places / Navigation managers.
/// Roads API posted-limit polling is disabled — OVER LIMIT UI uses Navigation SDK speeding %.
@MainActor
final class SpeedMonitor: NSObject, ObservableObject {
    @Published private(set) var speedMPH: Double?
    @Published private(set) var hasReliableSpeed = false
    /// Legacy Roads posted limit (unused by UI). Kept for now; no longer polled.
    @Published private(set) var postedSpeedLimitMPH: Int?

    // MARK: - Experimental tuning

    /// Reject updates whose horizontal speed accuracy is worse than this (meters/second).
    nonisolated static let maximumSpeedAccuracyMetersPerSecond: CLLocationSpeedAccuracy = 3.0

    /// EMA blend for readable but responsive MPH.
    nonisolated static let smoothingAlpha: Double = 0.25

    /// Below this smoothed MPH, display 0 to avoid GPS creep while stopped.
    nonisolated static let stationaryDisplayThresholdMPH: Double = 1.0

    /// Drop smoothing history if no valid fix arrives for this long.
    nonisolated static let smoothingResetGap: TimeInterval = 5.0

    /// Ignore location fixes older than this.
    nonisolated static let maximumLocationAge: TimeInterval = 5.0

    nonisolated static let metersPerSecondToMPH: Double = 2.2369362920544

    /// Horizontal accuracy gate for Roads path samples (meters).
    nonisolated static let maximumPathHorizontalAccuracyMeters: CLLocationAccuracy = 40.0

    /// Minimum distance between path samples (meters).
    nonisolated static let pathSampleMinSpacingMeters: CLLocationDistance = 12.0

    /// Max coordinates retained for each Roads request.
    nonisolated static let pathBufferCapacity = 12

    /// Minimum seconds between Roads speedLimits requests.
    nonisolated static let speedLimitRequestInterval: TimeInterval = 6.0

    /// Minimum travel since last successful limit before re-query (meters).
    nonisolated static let speedLimitMinTravelMeters: CLLocationDistance = 35.0

    private let manager = CLLocationManager()
    private var isMonitoring = false
    private var smoothedMPH: Double?
    private var lastValidMeasurementAt: Date?
    #if DEBUG
    private var lastAcceptedLogAt: Date?
    #endif

    private var pathSamples: [CLLocation] = []
    private var lastSpeedLimitRequestAt: Date?
    private var lastSpeedLimitCoordinate: CLLocationCoordinate2D?
    private var speedLimitTask: Task<Void, Never>?
    private var isSpeedLimitRequestInFlight = false

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBestForNavigation
        manager.activityType = .automotiveNavigation
        manager.distanceFilter = kCLDistanceFilterNone
        manager.pausesLocationUpdatesAutomatically = false
    }

    func start() {
        guard !isMonitoring else { return }
        isMonitoring = true

        switch manager.authorizationStatus {
        case .authorizedAlways, .authorizedWhenInUse:
            beginUpdating()
        case .notDetermined:
            manager.requestWhenInUseAuthorization()
        case .denied, .restricted:
            clearPublishedSpeed()
            clearPostedLimit()
        @unknown default:
            clearPublishedSpeed()
            clearPostedLimit()
        }
    }

    func stop() {
        guard isMonitoring else { return }
        isMonitoring = false
        manager.stopUpdatingLocation()
        smoothedMPH = nil
        lastValidMeasurementAt = nil
        speedLimitTask?.cancel()
        speedLimitTask = nil
        isSpeedLimitRequestInFlight = false
        pathSamples.removeAll()
        lastSpeedLimitRequestAt = nil
        lastSpeedLimitCoordinate = nil
        clearPublishedSpeed()
        clearPostedLimit()
    }

    private func beginUpdating() {
        manager.startUpdatingLocation()
    }

    private func clearPublishedSpeed() {
        speedMPH = nil
        hasReliableSpeed = false
    }

    private func clearPostedLimit() {
        postedSpeedLimitMPH = nil
    }

    // MARK: - Shared processing pipeline

    /// Validates and smooths a location fix. Used by live GPS and DEBUG injection.
    private func processLocation(_ location: CLLocation) {
        // Roads API speedLimits path sampling intentionally disabled (OVER LIMIT uses Nav SDK).

        switch validate(location) {
        case .rejected(let reason):
            #if DEBUG
            print("[Speed] rejected \(reason)")
            #endif
            return
        case .accepted(let mph):
            applySmoothing(
                rawMPH: mph,
                speedMetersPerSecond: location.speed,
                speedAccuracy: location.speedAccuracy,
                at: location.timestamp
            )
        }
    }

    private func handleLocations(_ locations: [CLLocation]) {
        guard isMonitoring, let location = locations.last else { return }
        processLocation(location)
    }

    private enum ValidationResult {
        case accepted(mph: Double)
        case rejected(String)
    }

    private func validate(_ location: CLLocation) -> ValidationResult {
        let age = -location.timestamp.timeIntervalSinceNow
        if age > Self.maximumLocationAge || age < -1 {
            return .rejected("stale age=\(String(format: "%.1f", age))s")
        }

        if location.speed < 0 {
            return .rejected("speed<0")
        }

        if location.speedAccuracy < 0 {
            return .rejected("speedAccuracy<0")
        }

        if location.speedAccuracy > Self.maximumSpeedAccuracyMetersPerSecond {
            return .rejected(
                String(format: "speedAccuracy=%.1f", location.speedAccuracy)
            )
        }

        let mph = location.speed * Self.metersPerSecondToMPH
        return .accepted(mph: mph)
    }

    private func applySmoothing(
        rawMPH: Double,
        speedMetersPerSecond: CLLocationSpeed,
        speedAccuracy: CLLocationSpeedAccuracy,
        at timestamp: Date
    ) {
        if let last = lastValidMeasurementAt,
           timestamp.timeIntervalSince(last) > Self.smoothingResetGap
            || last.timeIntervalSince(timestamp) > Self.smoothingResetGap {
            smoothedMPH = nil
        }
        lastValidMeasurementAt = timestamp

        let alpha = Self.smoothingAlpha
        let next: Double
        if let previous = smoothedMPH {
            next = alpha * rawMPH + (1.0 - alpha) * previous
        } else {
            next = rawMPH
        }
        smoothedMPH = next

        let display: Double
        if next < Self.stationaryDisplayThresholdMPH {
            display = 0
        } else {
            display = next
        }

        speedMPH = display
        hasReliableSpeed = true

        #if DEBUG
        logAcceptedIfNeeded(
            metersPerSecond: speedMetersPerSecond,
            accuracy: speedAccuracy,
            rawMPH: rawMPH,
            smoothed: next,
            at: timestamp
        )
        #endif
    }

    // MARK: - Google Roads posted limit

    private func ingestPathSample(_ location: CLLocation) {
        let age = -location.timestamp.timeIntervalSinceNow
        guard age <= Self.maximumLocationAge, age >= -1 else { return }
        guard location.horizontalAccuracy >= 0,
              location.horizontalAccuracy <= Self.maximumPathHorizontalAccuracyMeters else {
            return
        }
        guard CLLocationCoordinate2DIsValid(location.coordinate) else { return }

        if let last = pathSamples.last,
           last.distance(from: location) < Self.pathSampleMinSpacingMeters {
            return
        }

        pathSamples.append(location)
        if pathSamples.count > Self.pathBufferCapacity {
            pathSamples.removeFirst(pathSamples.count - Self.pathBufferCapacity)
        }
        requestSpeedLimitIfNeeded()
    }

    private func requestSpeedLimitIfNeeded() {
        guard isMonitoring, !isSpeedLimitRequestInFlight else { return }
        guard pathSamples.count >= 2 else { return }

        let now = Date()
        if let last = lastSpeedLimitRequestAt,
           now.timeIntervalSince(last) < Self.speedLimitRequestInterval {
            return
        }

        if let lastCoord = lastSpeedLimitCoordinate,
           let newest = pathSamples.last {
            let traveled = CLLocation(latitude: lastCoord.latitude, longitude: lastCoord.longitude)
                .distance(from: newest)
            if traveled < Self.speedLimitMinTravelMeters,
               postedSpeedLimitMPH != nil {
                return
            }
        }

        let coordinates = pathSamples.map(\.coordinate)
        isSpeedLimitRequestInFlight = true
        lastSpeedLimitRequestAt = now

        speedLimitTask?.cancel()
        speedLimitTask = Task { [weak self] in
            defer {
                Task { @MainActor in
                    self?.isSpeedLimitRequestInFlight = false
                }
            }
            do {
                let result = try await RoadsSpeedLimitService.shared.fetchSpeedLimitMPH(path: coordinates)
                await MainActor.run {
                    guard let self, self.isMonitoring else { return }
                    self.postedSpeedLimitMPH = result.speedLimitMPH
                    self.lastSpeedLimitCoordinate = coordinates.last
                    #if DEBUG
                    print("[SpeedLimit] Google Roads limit=\(result.speedLimitMPH) mph")
                    #endif
                }
            } catch {
                #if DEBUG
                print("[SpeedLimit] error: \(error.localizedDescription)")
                #endif
            }
        }
    }

    #if DEBUG
    private func logAcceptedIfNeeded(
        metersPerSecond: CLLocationSpeed,
        accuracy: CLLocationSpeedAccuracy,
        rawMPH: Double,
        smoothed: Double,
        at timestamp: Date
    ) {
        if let last = lastAcceptedLogAt, timestamp.timeIntervalSince(last) < 1.0 {
            return
        }
        lastAcceptedLogAt = timestamp
        print(
            String(
                format: "[Speed] raw=%.2fm/s accuracy=%.1fm/s mph=%.1f smoothed=%.1f",
                metersPerSecond,
                accuracy,
                rawMPH,
                smoothed
            )
        )
    }
    #endif
}

extension SpeedMonitor: CLLocationManagerDelegate {
    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            guard self.isMonitoring else { return }
            switch manager.authorizationStatus {
            case .authorizedAlways, .authorizedWhenInUse:
                self.beginUpdating()
            case .denied, .restricted:
                self.manager.stopUpdatingLocation()
                self.smoothedMPH = nil
                self.lastValidMeasurementAt = nil
                self.clearPublishedSpeed()
                self.clearPostedLimit()
            case .notDetermined:
                break
            @unknown default:
                break
            }
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        Task { @MainActor in
            self.handleLocations(locations)
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        #if DEBUG
        print("[Speed] location error: \(error.localizedDescription)")
        #endif
    }
}
