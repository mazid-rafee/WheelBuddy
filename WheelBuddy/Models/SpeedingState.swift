//
//  SpeedingState.swift
//  WheelBuddy
//

import Foundation

/// App-facing overspeed state derived from Google Navigation speed-alert callbacks.
/// Does not expose Google Navigation SDK types to SwiftUI.
enum SpeedingState: Equatable {
    /// No active guidance, or speed / posted limit unavailable.
    case unavailable
    /// Guidance active with a known limit; not speeding past configured thresholds.
    case normal
    /// Minor overspeed (caution).
    case minor
    /// Major overspeed (urgent).
    case major
}
