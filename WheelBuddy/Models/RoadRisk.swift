//
//  RoadRisk.swift
//  WheelBuddy
//

import Foundation

/// Experimental forward closing-risk estimation states (not validated FCW).
enum RoadRiskState: Equatable {
    case clear
    case monitoring
    case caution
    case high
}

struct RoadRiskResult: Equatable {
    let state: RoadRiskState
    let leadVehicle: RoadDetection?
    /// Relative apparent-scale growth per second, if available.
    let relativeExpansionRate: Double?
    let trackAge: TimeInterval
}
