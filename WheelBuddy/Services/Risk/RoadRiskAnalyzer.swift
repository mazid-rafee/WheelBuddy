//
//  RoadRiskAnalyzer.swift
//  WheelBuddy
//
//  Experimental monocular forward closing-risk estimation.
//  Consumes [RoadDetection] only — no camera frames, no Core ML, no TTC/distance claims.
//

import Combine
import CoreGraphics
import Foundation

/// Estimates whether a likely lead vehicle ahead shows sustained apparent closing motion.
@MainActor
final class RoadRiskAnalyzer: ObservableObject {
    @Published private(set) var state: RoadRiskState = .clear
    @Published private(set) var leadVehicle: RoadDetection?
    @Published private(set) var lastResult = RoadRiskResult(
        state: .clear,
        leadVehicle: nil,
        relativeExpansionRate: nil,
        trackAge: 0
    )

    // MARK: - EXPERIMENTAL / INITIAL TUNING VALUES

    private enum Config {
        /// Motor-vehicle classes eligible as lead (person/bicycle excluded).
        static let vehicleLabels: Set<String> = ["car", "truck", "bus", "motorcycle"]

        /// Minimum normalized box area (~0.25% of image).
        static let minimumVehicleArea: CGFloat = 0.0025

        /// Max |midX - 0.5| for corridor eligibility.
        static let maxHorizontalOffset: CGFloat = 0.35

        /// Lead score weights.
        static let centralityWeight: Double = 0.65
        static let areaWeight: Double = 0.35
        /// Area score saturates near this normalized area.
        static let areaScoreReference: CGFloat = 0.12

        /// Temporal association.
        static let minimumIoU: CGFloat = 0.20
        static let fallbackCenterDistance: CGFloat = 0.15
        static let fallbackMinAreaRatio: CGFloat = 0.5
        static let fallbackMaxAreaRatio: CGFloat = 2.0

        /// Drop track if no association for this long.
        static let trackLossTimeout: TimeInterval = 0.75

        /// Bounded history (detector ~5 FPS).
        static let maxHistoryCount = 8

        /// EMA smoothing on apparent scale.
        static let scaleSmoothingAlpha: Double = 0.35

        /// Prefer expansion over ~0.6–1.5 s.
        static let expansionLookbackMin: TimeInterval = 0.6
        static let expansionLookbackMax: TimeInterval = 1.5

        /// Minimum track age before caution/high.
        static let minimumRiskTrackAge: TimeInterval = 0.6

        /// CAUTION thresholds.
        /// Empirically adjusted from iPhone 13 mini RoadRiskDebug runs (static ~−0.04…+0.004/s,
        /// sustained closing ~0.07…0.12/s); remain experimental — not validated FCW thresholds.
        static let cautionMinArea: CGFloat = 0.01
        static let cautionMinExpansionPerSec: Double = 0.06
        static let cautionMinCentrality: Double = 0.10

        /// HIGH thresholds.
        static let highMinArea: CGFloat = 0.025
        static let highMinExpansionPerSec: Double = 0.25
        static let highMinCentrality: Double = 0.55

        /// Persistence before elevating.
        static let cautionPersistCount = 2
        static let highPersistCount = 3

        /// Hysteresis: safer updates required before downgrade.
        static let cautionDowngradeSafeCount = 2
        static let highDowngradeSafeCount = 3

        /// Temporary periodic lead-track diagnostic (not every update).
        static let diagnosticEveryNUpdates = 5
    }

    // MARK: - Track state

    private struct HistorySample {
        let timestamp: TimeInterval
        let boundingBox: CGRect
        let area: CGFloat
        let smoothedScale: Double
    }

    private var trackHistory: [HistorySample] = []
    private var trackStartedAt: TimeInterval?
    private var lastAssociatedAt: TimeInterval?
    private var currentLead: RoadDetection?

    private var cautionStreak = 0
    private var highStreak = 0
    private var saferStreak = 0

    private var updateCount = 0
    /// Throttle for size-inconsistency association rejects.
    private var lastSizeRejectLogUpdate = 0

    // MARK: - Public API

    func update(detections: [RoadDetection], timestamp: TimeInterval) {
        updateCount += 1

        // Drop stale track.
        if let last = lastAssociatedAt,
           timestamp - last >= Config.trackLossTimeout {
            clearTrack(reason: "track loss timeout")
        }

        let candidates = vehicleCandidates(from: detections)

        if let lead = currentLead {
            if let match = associateLead(previous: lead, candidates: candidates) {
                appendObservation(match, timestamp: timestamp)
            } else {
                // Missing this update — do not attach unrelated vehicle yet.
                publish(expansion: nil, timestamp: timestamp)
                return
            }
        } else if let initial = selectInitialLead(from: candidates) {
            beginTrack(with: initial, timestamp: timestamp)
        } else {
            publishClear(timestamp: timestamp)
            return
        }

        let expansion = calculateRelativeExpansion(at: timestamp)
        let proposed = evaluateRiskLevel(
            expansion: expansion,
            timestamp: timestamp
        )
        applyPersistenceAndHysteresis(proposed: proposed)
        publish(expansion: expansion, timestamp: timestamp)
        maybeLogDiagnostic(expansion: expansion)
    }

    func reset() {
        clearTrack(reason: nil)
        cautionStreak = 0
        highStreak = 0
        saferStreak = 0
        updateCount = 0
        lastSizeRejectLogUpdate = 0
        state = .clear
        leadVehicle = nil
        lastResult = RoadRiskResult(state: .clear, leadVehicle: nil, relativeExpansionRate: nil, trackAge: 0)
    }

    // MARK: - Candidates & selection

    private func vehicleCandidates(from detections: [RoadDetection]) -> [RoadDetection] {
        detections.filter { detection in
            guard Config.vehicleLabels.contains(detection.label.lowercased()) else { return false }
            let area = boxArea(detection.boundingBox)
            guard area >= Config.minimumVehicleArea else { return false }
            let offset = abs(detection.boundingBox.midX - 0.5)
            return offset <= Config.maxHorizontalOffset
        }
    }

    private func selectInitialLead(from candidates: [RoadDetection]) -> RoadDetection? {
        candidates.max { a, b in
            leadScore(a) < leadScore(b)
        }
    }

    private func leadScore(_ detection: RoadDetection) -> Double {
        let area = boxArea(detection.boundingBox)
        let offset = abs(detection.boundingBox.midX - 0.5)
        let centrality = 1.0 - min(Double(offset / Config.maxHorizontalOffset), 1.0)
        let areaScore = min(Double(area / Config.areaScoreReference), 1.0)
        return Config.centralityWeight * centrality + Config.areaWeight * areaScore
    }

    private func centrality(of box: CGRect) -> Double {
        let offset = abs(box.midX - 0.5)
        return 1.0 - min(Double(offset / Config.maxHorizontalOffset), 1.0)
    }

    // MARK: - Association

    /// Associate a new detection with the current lead.
    /// Requires (IoU ≥ minimumIoU OR fallback center-distance) AND area ratio in
    /// [fallbackMinAreaRatio, fallbackMaxAreaRatio]. High IoU alone cannot attach a
    /// dramatically different-sized box (likely a different vehicle).
    private func associateLead(previous: RoadDetection, candidates: [RoadDetection]) -> RoadDetection? {
        let prevArea = max(boxArea(previous.boundingBox), 1e-9)

        var bestIoU: CGFloat = 0
        var bestIoUMatch: RoadDetection?
        var sizeRejectSample: (ratio: CGFloat, iou: CGFloat, centerDist: CGFloat)?

        for candidate in candidates {
            let iou = Self.intersectionOverUnion(previous.boundingBox, candidate.boundingBox)
            let dist = centerDistance(previous.boundingBox, candidate.boundingBox)
            let ratio = boxArea(candidate.boundingBox) / prevArea
            let sizeOK = ratio >= Config.fallbackMinAreaRatio && ratio <= Config.fallbackMaxAreaRatio

            if iou >= Config.minimumIoU {
                if sizeOK {
                    if iou > bestIoU {
                        bestIoU = iou
                        bestIoUMatch = candidate
                    }
                } else if sizeRejectSample == nil || iou > sizeRejectSample!.iou {
                    sizeRejectSample = (ratio, iou, dist)
                }
            }
        }

        if let match = bestIoUMatch {
            return match
        }

        // Fallback: center proximity + size consistency (same area-ratio gate).
        var bestDist = CGFloat.greatestFiniteMagnitude
        var bestFallback: RoadDetection?

        for candidate in candidates {
            let dist = centerDistance(previous.boundingBox, candidate.boundingBox)
            guard dist < Config.fallbackCenterDistance else { continue }
            let ratio = boxArea(candidate.boundingBox) / prevArea
            if ratio >= Config.fallbackMinAreaRatio, ratio <= Config.fallbackMaxAreaRatio {
                if dist < bestDist {
                    bestDist = dist
                    bestFallback = candidate
                }
            } else if sizeRejectSample == nil {
                let iou = Self.intersectionOverUnion(previous.boundingBox, candidate.boundingBox)
                sizeRejectSample = (ratio, iou, dist)
            }
        }

        if let match = bestFallback {
            return match
        }

        if let sample = sizeRejectSample {
            logSizeRejectThrottled(areaRatio: sample.ratio, iou: sample.iou, centerDist: sample.centerDist)
        }
        return nil
    }

    private func logSizeRejectThrottled(areaRatio: CGFloat, iou: CGFloat, centerDist: CGFloat) {
        // Avoid spamming Console; roughly align with RoadRiskDebug cadence.
        guard updateCount - lastSizeRejectLogUpdate >= 5 else { return }
        lastSizeRejectLogUpdate = updateCount
        print(
            String(
                format: "[RoadRiskDebug] rejected association areaRatio=%.3f iou=%.3f centerDist=%.3f",
                Double(areaRatio),
                Double(iou),
                Double(centerDist)
            )
        )
    }

    // MARK: - Track history

    private func beginTrack(with detection: RoadDetection, timestamp: TimeInterval) {
        let area = boxArea(detection.boundingBox)
        let scale = apparentScale(area: area)
        currentLead = detection
        trackStartedAt = timestamp
        lastAssociatedAt = timestamp
        trackHistory = [
            HistorySample(
                timestamp: timestamp,
                boundingBox: detection.boundingBox,
                area: area,
                smoothedScale: scale
            )
        ]
        cautionStreak = 0
        highStreak = 0
        saferStreak = 0
        transitionLog(from: state, to: .monitoring, detail: "lead=\(detection.label) area=\(String(format: "%.3f", Double(area))) centrality=\(String(format: "%.2f", centrality(of: detection.boundingBox)))")
        state = .monitoring
    }

    private func appendObservation(_ detection: RoadDetection, timestamp: TimeInterval) {
        let area = boxArea(detection.boundingBox)
        let rawScale = apparentScale(area: area)
        let previous = trackHistory.last?.smoothedScale ?? rawScale
        let smoothed = Config.scaleSmoothingAlpha * rawScale
            + (1.0 - Config.scaleSmoothingAlpha) * previous

        currentLead = detection
        lastAssociatedAt = timestamp
        trackHistory.append(
            HistorySample(
                timestamp: timestamp,
                boundingBox: detection.boundingBox,
                area: area,
                smoothedScale: smoothed
            )
        )
        if trackHistory.count > Config.maxHistoryCount {
            trackHistory.removeFirst(trackHistory.count - Config.maxHistoryCount)
        }
    }

    private func clearTrack(reason: String?) {
        if let reason, currentLead != nil {
            print("[RoadRisk] track cleared: \(reason)")
        }
        trackHistory.removeAll()
        trackStartedAt = nil
        lastAssociatedAt = nil
        currentLead = nil
        cautionStreak = 0
        highStreak = 0
        saferStreak = 0
    }

    // MARK: - Expansion & risk

    private func calculateRelativeExpansion(at timestamp: TimeInterval) -> Double? {
        guard let newest = trackHistory.last else { return nil }

        // Prefer sample ~0.6–1.5 s earlier.
        let olderCandidates = trackHistory.filter { sample in
            let dt = newest.timestamp - sample.timestamp
            return dt >= Config.expansionLookbackMin && dt <= Config.expansionLookbackMax
        }

        let older: HistorySample?
        if let preferred = olderCandidates.min(by: {
            abs(($0.timestamp) - (newest.timestamp - 1.0)) < abs(($1.timestamp) - (newest.timestamp - 1.0))
        }) {
            older = preferred
        } else {
            // Fall back to oldest if we have enough span but outside ideal window.
            older = trackHistory.first.flatMap { first in
                let dt = newest.timestamp - first.timestamp
                return dt >= Config.expansionLookbackMin ? first : nil
            }
        }

        guard let old = older else { return nil }
        let deltaTime = newest.timestamp - old.timestamp
        guard deltaTime > 1e-3 else { return nil }
        guard old.smoothedScale > 1e-6 else { return nil }

        let rate = (newest.smoothedScale - old.smoothedScale) / old.smoothedScale / deltaTime
        guard rate.isFinite else { return nil }
        return rate
    }

    private enum ProposedLevel {
        case clear
        case monitoring
        case caution
        case high
    }

    private func evaluateRiskLevel(expansion: Double?, timestamp: TimeInterval) -> ProposedLevel {
        guard let lead = currentLead, let started = trackStartedAt else {
            return .clear
        }

        let age = timestamp - started
        let area = boxArea(lead.boundingBox)
        let cent = centrality(of: lead.boundingBox)

        guard let expansion, age >= Config.minimumRiskTrackAge else {
            return .monitoring
        }

        let highReady = area >= Config.highMinArea
            && expansion >= Config.highMinExpansionPerSec
            && cent >= Config.highMinCentrality

        if highReady {
            return .high
        }

        let cautionReady = area >= Config.cautionMinArea
            && expansion >= Config.cautionMinExpansionPerSec
            && cent >= Config.cautionMinCentrality

        if cautionReady {
            return .caution
        }

        return .monitoring
    }

    private func applyPersistenceAndHysteresis(proposed: ProposedLevel) {
        let previous = state

        switch proposed {
        case .high:
            highStreak += 1
            cautionStreak += 1
            saferStreak = 0
        case .caution:
            cautionStreak += 1
            highStreak = 0
            saferStreak = 0
        case .monitoring, .clear:
            highStreak = 0
            cautionStreak = 0
            saferStreak += 1
        }

        var next = previous

        switch previous {
        case .clear, .monitoring:
            if proposed == .high, highStreak >= Config.highPersistCount {
                next = .high
            } else if (proposed == .caution || proposed == .high),
                      cautionStreak >= Config.cautionPersistCount {
                next = .caution
            } else if currentLead != nil {
                next = .monitoring
            } else {
                next = .clear
            }

        case .caution:
            if proposed == .high, highStreak >= Config.highPersistCount {
                next = .high
            } else if proposed == .monitoring || proposed == .clear {
                if saferStreak >= Config.cautionDowngradeSafeCount {
                    next = currentLead == nil ? .clear : .monitoring
                }
            }

        case .high:
            if proposed == .high {
                next = .high
            } else if saferStreak >= Config.highDowngradeSafeCount {
                if proposed == .caution, cautionStreak >= Config.cautionPersistCount {
                    next = .caution
                } else {
                    next = currentLead == nil ? .clear : .monitoring
                }
            }
        }

        if next != previous {
            let expansionNote: String
            if let newest = trackHistory.last {
                expansionNote = "area=\(String(format: "%.3f", Double(newest.area)))"
            } else {
                expansionNote = ""
            }
            transitionLog(from: previous, to: next, detail: expansionNote)
        }
        state = next
    }

    // MARK: - Publish

    private func publish(expansion: Double?, timestamp: TimeInterval) {
        let age: TimeInterval
        if let started = trackStartedAt {
            age = max(0, timestamp - started)
        } else {
            age = 0
        }

        if currentLead == nil {
            state = .clear
        }

        leadVehicle = currentLead
        lastResult = RoadRiskResult(
            state: state,
            leadVehicle: currentLead,
            relativeExpansionRate: expansion,
            trackAge: age
        )
    }

    private func publishClear(timestamp: TimeInterval) {
        let previous = state
        clearTrack(reason: nil)
        if previous != .clear {
            transitionLog(from: previous, to: .clear, detail: "no lead candidate")
        }
        state = .clear
        leadVehicle = nil
        lastResult = RoadRiskResult(state: .clear, leadVehicle: nil, relativeExpansionRate: nil, trackAge: 0)
    }

    private func maybeLogDiagnostic(expansion: Double?) {
        guard let lead = currentLead, updateCount % Config.diagnosticEveryNUpdates == 0 else { return }

        let area = boxArea(lead.boundingBox)
        let scale = trackHistory.last?.smoothedScale ?? apparentScale(area: area)
        let expText = expansion.map { String(format: "%.3f/s", $0) } ?? "n/a"
        let cent = centrality(of: lead.boundingBox)
        let age: TimeInterval
        if let started = trackStartedAt, let last = lastAssociatedAt {
            age = max(0, last - started)
        } else {
            age = 0
        }

        print(
            String(
                format: "[RoadRiskDebug] area=%.4f scale=%.4f exp=%@ centrality=%.2f age=%.2f state=%@",
                Double(area),
                scale,
                expText,
                cent,
                age,
                String(describing: state)
            )
        )
    }

    private func transitionLog(from: RoadRiskState, to: RoadRiskState, detail: String) {
        let suffix = detail.isEmpty ? "" : " \(detail)"
        print("[RoadRisk] \(label(from)) -> \(label(to))\(suffix)")
    }

    private func label(_ state: RoadRiskState) -> String {
        switch state {
        case .clear: return "CLEAR"
        case .monitoring: return "MONITORING"
        case .caution: return "CAUTION"
        case .high: return "HIGH"
        }
    }

    // MARK: - Geometry helpers

    private func boxArea(_ box: CGRect) -> CGFloat {
        max(0, box.width) * max(0, box.height)
    }

    private func apparentScale(area: CGFloat) -> Double {
        Double(sqrt(max(area, 0)))
    }

    private func centerDistance(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let dx = a.midX - b.midX
        let dy = a.midY - b.midY
        return sqrt(dx * dx + dy * dy)
    }

    nonisolated static func intersectionOverUnion(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        guard !intersection.isNull else { return 0 }

        let interArea = max(0, intersection.width) * max(0, intersection.height)
        let areaA = max(0, a.width) * max(0, a.height)
        let areaB = max(0, b.width) * max(0, b.height)
        let union = areaA + areaB - interArea
        guard union > 1e-12 else { return 0 }
        return interArea / union
    }
}
