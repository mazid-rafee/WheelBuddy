//
//  LaneAssistState.swift
//  WheelBuddy
//

import CoreGraphics
import Foundation

/// Experimental lane-assist state — independent of driver attention / road risk.
enum LaneAssistState: Equatable {
    case unavailable
    case tracking
    case driftingLeft
    case driftingRight
}

/// One frame’s smoothed lane-tracking telemetry for UI / alerts.
struct LaneTrackingResult: Equatable {
    let state: LaneAssistState

    /// Normalized lateral position: −1 ≈ left boundary, 0 = centered, +1 ≈ right boundary.
    /// Negative = vehicle toward LEFT of lane; positive = toward RIGHT.
    let lateralOffset: CGFloat

    let confidence: CGFloat

    /// Normalized image X (top-left origin, 0…1) at the near-field evaluation row.
    let leftLaneBottomX: CGFloat?
    let rightLaneBottomX: CGFloat?
    let laneCenterX: CGFloat?

    static let empty = LaneTrackingResult(
        state: .unavailable,
        lateralOffset: 0,
        confidence: 0,
        leftLaneBottomX: nil,
        rightLaneBottomX: nil,
        laneCenterX: nil
    )
}

/// Frame-level lane perception (no EMA / speed / drift). Coordinates are normalized top-left (y=0 top).
struct LanePerceptionResult: Sendable {
    let leftLanePoints: [CGPoint]
    let rightLanePoints: [CGPoint]
    let leftXAtEvaluationY: CGFloat?
    let rightXAtEvaluationY: CGFloat?
    /// Frame-level score; same CGFloat convention as `LaneTrackingResult.confidence`.
    let confidence: CGFloat

    static let empty = LanePerceptionResult(
        leftLanePoints: [],
        rightLanePoints: [],
        leftXAtEvaluationY: nil,
        rightXAtEvaluationY: nil,
        confidence: 0
    )
}

#if DEBUG
/// Extra geometry for the DEBUG lane overlay (not used in production UI).
struct LaneDebugSnapshot: Equatable {
    var roiTopY: CGFloat = 0.45
    var roiBottomY: CGFloat = 0.95
    var roiTopLeftX: CGFloat = 0.35
    var roiTopRightX: CGFloat = 0.65
    var roiBottomLeftX: CGFloat = 0.05
    var roiBottomRightX: CGFloat = 0.95
    var evaluationY: CGFloat = 0.85
    var leftX: CGFloat?
    var rightX: CGFloat?
    var laneCenterX: CGFloat?
    var vehicleCenterX: CGFloat = 0.50
    var confidence: CGFloat = 0
    var lateralOffset: CGFloat = 0
    /// Fitted polylines from the current backend frame (normalized top-left).
    var leftLanePoints: [CGPoint] = []
    var rightLanePoints: [CGPoint] = []
    /// Unsmoothed backend confidence for this frame.
    var frameConfidence: CGFloat = 0
}
#endif
