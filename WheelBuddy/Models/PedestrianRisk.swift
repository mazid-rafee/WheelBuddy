//
//  PedestrianRisk.swift
//  WheelBuddy
//
//  Visible pedestrian ROAD states: ahead (informational) and close (critical).
//  Corridor membership is an internal cue only — never a visible UI state.
//

import CoreGraphics
import Foundation

/// Visible pedestrian risk levels for the unified ROAD status.
enum PedestrianRiskState: Equatable {
    /// No relevant person tracked.
    case clear
    /// Person relevant to the road scene; not an immediate close conflict.
    case ahead
    /// High visual collision risk (corridor + proximity/expansion + persistence).
    case close
}

struct PedestrianRiskResult: Equatable {
    let state: PedestrianRiskState
    let leadPerson: RoadDetection?
    let relativeExpansionRate: Double?
    let trackAge: TimeInterval
    /// Internal: bottom-center inside/near ego corridor (not a UI state).
    let isInCorridor: Bool

    static let empty = PedestrianRiskResult(
        state: .clear,
        leadPerson: nil,
        relativeExpansionRate: nil,
        trackAge: 0,
        isInCorridor: false
    )
}

/// Unified presentation for the single ROAD chip / banner / alerts.
enum UnifiedRoadDisplayState: Equatable {
    case unavailable
    case clear
    case vehicleAhead
    case closingVehicle
    case rapidClosing
    case pedestrianAhead
    case pedestrianClose

    var statusText: String {
        switch self {
        case .unavailable:
            return "Road monitoring unavailable"
        case .clear:
            return "Clear"
        case .vehicleAhead:
            return "Vehicle ahead"
        case .closingVehicle:
            return "Closing vehicle"
        case .rapidClosing:
            return "Rapid closing"
        case .pedestrianAhead:
            return "Pedestrian ahead"
        case .pedestrianClose:
            return "Pedestrian Close!"
        }
    }

    /// SF Symbol for the existing ROAD icon slot.
    var iconName: String {
        switch self {
        case .pedestrianAhead, .pedestrianClose:
            return "figure.walk"
        case .unavailable, .clear, .vehicleAhead, .closingVehicle, .rapidClosing:
            return "car.fill"
        }
    }

    /// Priority rank (higher = more urgent for display).
    var displayPriority: Int {
        switch self {
        case .pedestrianClose: return 6
        case .rapidClosing: return 5
        case .closingVehicle: return 4
        case .pedestrianAhead: return 3
        case .vehicleAhead: return 2
        case .clear: return 1
        case .unavailable: return 0
        }
    }

    static func resolve(
        modelReady: Bool,
        modelUnavailable: Bool,
        vehicle: RoadRiskState,
        pedestrian: PedestrianRiskState
    ) -> UnifiedRoadDisplayState {
        if !modelReady || modelUnavailable {
            return .unavailable
        }

        let vehicleDisplay: UnifiedRoadDisplayState
        switch vehicle {
        case .clear:
            vehicleDisplay = .clear
        case .monitoring:
            vehicleDisplay = .vehicleAhead
        case .caution:
            vehicleDisplay = .closingVehicle
        case .high:
            vehicleDisplay = .rapidClosing
        }

        let pedestrianDisplay: UnifiedRoadDisplayState
        switch pedestrian {
        case .clear:
            pedestrianDisplay = .clear
        case .ahead:
            pedestrianDisplay = .pedestrianAhead
        case .close:
            pedestrianDisplay = .pedestrianClose
        }

        if pedestrianDisplay.displayPriority >= vehicleDisplay.displayPriority {
            return pedestrianDisplay
        }
        return vehicleDisplay
    }
}
