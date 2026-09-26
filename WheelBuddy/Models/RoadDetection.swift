//
//  RoadDetection.swift
//  WheelBuddy
//

import CoreGraphics
import Foundation

struct RoadDetection: Identifiable, Equatable {
    let id: UUID
    let label: String
    let confidence: Float
    /// Vision normalized bounding box (origin bottom-left).
    let boundingBox: CGRect

    init(
        id: UUID = UUID(),
        label: String,
        confidence: Float,
        boundingBox: CGRect
    ) {
        self.id = id
        self.label = label
        self.confidence = confidence
        self.boundingBox = boundingBox
    }
}

enum RoadMonitoringState: Equatable {
    case clear
    case objectsDetected
    case modelUnavailable
}
