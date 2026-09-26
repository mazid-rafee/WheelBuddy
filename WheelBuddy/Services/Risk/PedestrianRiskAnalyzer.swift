//
//  PedestrianRiskAnalyzer.swift
//  WheelBuddy
//
//  Experimental monocular pedestrian risk for the unified ROAD status.
//  Consumes [RoadDetection] person labels only — independent of RoadRiskAnalyzer.
//  Visible states: ahead | close. Corridor is an internal cue for close eligibility.
//

import Combine
import CoreGraphics
import Foundation

/// Tracks person detections and publishes ahead / close pedestrian risk.
@MainActor
final class PedestrianRiskAnalyzer: ObservableObject {
    @Published private(set) var state: PedestrianRiskState = .clear
    @Published private(set) var leadPerson: RoadDetection?
    @Published private(set) var lastResult = PedestrianRiskResult.empty

    // MARK: - EXPERIMENTAL / INITIAL TUNING (detector ~5 FPS)

    private enum Config {
        // Relevance (Pedestrian ahead) — broader than close corridor.
        static let minimumPersonArea: CGFloat = 0.004
        static let maxRelevanceHorizontalOffset: CGFloat = 0.42
        /// Ignore people whose bottom edge is still very high (far / overhead).
        static let minRelevanceBottomY: CGFloat = 0.20

        // Ego corridor (Vision normalized, origin bottom-left).
        // Narrower toward horizon (high y), wider toward bottom (low y).
        static let corridorFarY: CGFloat = 0.72
        static let corridorNearY: CGFloat = 0.08
        static let corridorFarHalfWidth: CGFloat = 0.10
        static let corridorNearHalfWidth: CGFloat = 0.28
        /// Extra half-width margin for “very near” corridor eligibility.
        static let corridorNearMargin: CGFloat = 0.04
        static let corridorCenterX: CGFloat = 0.50

        // Association (~5 Hz).
        static let minimumIoU: CGFloat = 0.15
        static let fallbackCenterDistance: CGFloat = 0.18
        static let fallbackMinAreaRatio: CGFloat = 0.45
        static let fallbackMaxAreaRatio: CGFloat = 2.2
        static let trackLossTimeout: TimeInterval = 0.90
        static let maxHistoryCount = 10
        static let scaleSmoothingAlpha: Double = 0.35

        // Close risk cues.
        static let minimumCloseTrackAge: TimeInterval = 0.55
        static let closeMinArea: CGFloat = 0.035
        static let closeMinExpansionPerSec: Double = 0.18
        static let expansionLookbackMin: TimeInterval = 0.50
        static let expansionLookbackMax: TimeInterval = 1.40

        // Persistence / hysteresis (frames at ~5 Hz).
        static let aheadPersistCount = 2
        static let closePersistCount = 3
        static let closeDowngradeSafeCount = 3
        static let aheadDowngradeSafeCount = 2

        static let diagnosticEveryNUpdates = 5
    }

    private struct HistorySample {
        let timestamp: TimeInterval
        let boundingBox: CGRect
        let area: CGFloat
        let smoothedScale: Double
        let bottomCenter: CGPoint
        let inCorridor: Bool
    }

    private var trackHistory: [HistorySample] = []
    private var trackStartedAt: TimeInterval?
    private var lastAssociatedAt: TimeInterval?
    private var currentLead: RoadDetection?
    private var consecutiveHits = 0
    private var missingFrames = 0
    private var currentlyInCorridor = false

    private var aheadStreak = 0
    private var closeStreak = 0
    private var saferStreak = 0

    private var updateCount = 0

    // MARK: - Public API

    func update(detections: [RoadDetection], timestamp: TimeInterval) {
        updateCount += 1

        if let last = lastAssociatedAt,
           timestamp - last >= Config.trackLossTimeout {
            clearTrack(reason: "track loss timeout")
        }

        let candidates = personCandidates(from: detections)

        if let lead = currentLead {
            if let match = associateLead(previous: lead, candidates: candidates) {
                appendObservation(match, timestamp: timestamp)
                missingFrames = 0
                consecutiveHits += 1
            } else {
                missingFrames += 1
                // Keep last published state briefly to avoid ahead/clear flicker.
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
        let proposed = evaluateRiskLevel(expansion: expansion, timestamp: timestamp)
        applyPersistenceAndHysteresis(proposed: proposed)
        publish(expansion: expansion, timestamp: timestamp)
        maybeLogDiagnostic(expansion: expansion)
    }

    func reset() {
        clearTrack(reason: nil)
        aheadStreak = 0
        closeStreak = 0
        saferStreak = 0
        updateCount = 0
        state = .clear
        leadPerson = nil
        lastResult = .empty
    }

    // MARK: - Candidates

    private func personCandidates(from detections: [RoadDetection]) -> [RoadDetection] {
        detections.filter { detection in
            guard detection.label.lowercased() == "person" else { return false }
            let area = boxArea(detection.boundingBox)
            guard area >= Config.minimumPersonArea else { return false }
            let bottom = bottomCenter(of: detection.boundingBox)
            guard bottom.y >= Config.minRelevanceBottomY else { return false }
            let offset = abs(detection.boundingBox.midX - 0.5)
            return offset <= Config.maxRelevanceHorizontalOffset
        }
    }

    private func selectInitialLead(from candidates: [RoadDetection]) -> RoadDetection? {
        candidates.max { a, b in
            personScore(a) < personScore(b)
        }
    }

    private func personScore(_ detection: RoadDetection) -> Double {
        let area = boxArea(detection.boundingBox)
        let bottom = bottomCenter(of: detection.boundingBox)
        let inCorridor = isBottomCenterInCorridor(bottom, includeMargin: true)
        let offset = abs(bottom.x - Config.corridorCenterX)
        let centrality = 1.0 - min(Double(offset / Config.corridorNearHalfWidth), 1.0)
        let areaScore = min(Double(area / 0.08), 1.0)
        let corridorBoost = inCorridor ? 0.35 : 0.0
        return 0.45 * centrality + 0.35 * areaScore + corridorBoost
            + 0.20 * Double(detection.confidence)
    }

    // MARK: - Association

    private func associateLead(previous: RoadDetection, candidates: [RoadDetection]) -> RoadDetection? {
        let prevArea = max(boxArea(previous.boundingBox), 1e-9)

        var bestIoU: CGFloat = 0
        var bestIoUMatch: RoadDetection?

        for candidate in candidates {
            let iou = RoadRiskAnalyzer.intersectionOverUnion(previous.boundingBox, candidate.boundingBox)
            let ratio = boxArea(candidate.boundingBox) / prevArea
            let sizeOK = ratio >= Config.fallbackMinAreaRatio && ratio <= Config.fallbackMaxAreaRatio
            if iou >= Config.minimumIoU, sizeOK, iou > bestIoU {
                bestIoU = iou
                bestIoUMatch = candidate
            }
        }
        if let match = bestIoUMatch {
            return match
        }

        var bestDist = CGFloat.greatestFiniteMagnitude
        var bestFallback: RoadDetection?
        for candidate in candidates {
            let dist = centerDistance(previous.boundingBox, candidate.boundingBox)
            guard dist < Config.fallbackCenterDistance else { continue }
            let ratio = boxArea(candidate.boundingBox) / prevArea
            guard ratio >= Config.fallbackMinAreaRatio, ratio <= Config.fallbackMaxAreaRatio else { continue }
            if dist < bestDist {
                bestDist = dist
                bestFallback = candidate
            }
        }
        return bestFallback
    }

    // MARK: - Track

    private func beginTrack(with detection: RoadDetection, timestamp: TimeInterval) {
        let area = boxArea(detection.boundingBox)
        let bottom = bottomCenter(of: detection.boundingBox)
        let inCorridor = isBottomCenterInCorridor(bottom, includeMargin: true)
        currentLead = detection
        trackStartedAt = timestamp
        lastAssociatedAt = timestamp
        consecutiveHits = 1
        missingFrames = 0
        currentlyInCorridor = inCorridor
        trackHistory = [
            HistorySample(
                timestamp: timestamp,
                boundingBox: detection.boundingBox,
                area: area,
                smoothedScale: apparentScale(area: area),
                bottomCenter: bottom,
                inCorridor: inCorridor
            )
        ]
        aheadStreak = 0
        closeStreak = 0
        saferStreak = 0
        // Do not jump to ahead until persistence — still publish monitoring path via evaluate.
    }

    private func appendObservation(_ detection: RoadDetection, timestamp: TimeInterval) {
        let area = boxArea(detection.boundingBox)
        let rawScale = apparentScale(area: area)
        let previous = trackHistory.last?.smoothedScale ?? rawScale
        let smoothed = Config.scaleSmoothingAlpha * rawScale
            + (1.0 - Config.scaleSmoothingAlpha) * previous
        let bottom = bottomCenter(of: detection.boundingBox)
        let inCorridor = isBottomCenterInCorridor(bottom, includeMargin: true)

        currentLead = detection
        lastAssociatedAt = timestamp
        currentlyInCorridor = inCorridor
        trackHistory.append(
            HistorySample(
                timestamp: timestamp,
                boundingBox: detection.boundingBox,
                area: area,
                smoothedScale: smoothed,
                bottomCenter: bottom,
                inCorridor: inCorridor
            )
        )
        if trackHistory.count > Config.maxHistoryCount {
            trackHistory.removeFirst(trackHistory.count - Config.maxHistoryCount)
        }
    }

    private func clearTrack(reason: String?) {
        if let reason, currentLead != nil {
            #if DEBUG
            print("[PedestrianRisk] track cleared: \(reason)")
            #endif
        }
        trackHistory.removeAll()
        trackStartedAt = nil
        lastAssociatedAt = nil
        currentLead = nil
        consecutiveHits = 0
        missingFrames = 0
        currentlyInCorridor = false
        aheadStreak = 0
        closeStreak = 0
        saferStreak = 0
    }

    // MARK: - Corridor (Vision bottom-left)

    /// Bottom-center contact point: midX, minY (Vision origin is bottom-left).
    private func bottomCenter(of box: CGRect) -> CGPoint {
        CGPoint(x: box.midX, y: box.minY)
    }

    private func corridorHalfWidth(atY y: CGFloat) -> CGFloat {
        let clampedY = min(max(y, Config.corridorNearY), Config.corridorFarY)
        let span = max(Config.corridorFarY - Config.corridorNearY, 1e-6)
        // t=0 at near (bottom), t=1 at far (top).
        let t = (clampedY - Config.corridorNearY) / span
        return Config.corridorNearHalfWidth
            + (Config.corridorFarHalfWidth - Config.corridorNearHalfWidth) * t
    }

    private func isBottomCenterInCorridor(_ point: CGPoint, includeMargin: Bool) -> Bool {
        guard point.y >= Config.corridorNearY - 0.02,
              point.y <= Config.corridorFarY + 0.05 else {
            return false
        }
        let half = corridorHalfWidth(atY: point.y)
            + (includeMargin ? Config.corridorNearMargin : 0)
        return abs(point.x - Config.corridorCenterX) <= half
    }

    // MARK: - Expansion & risk

    private func calculateRelativeExpansion(at timestamp: TimeInterval) -> Double? {
        guard let newest = trackHistory.last else { return nil }

        let olderCandidates = trackHistory.filter { sample in
            let dt = newest.timestamp - sample.timestamp
            return dt >= Config.expansionLookbackMin && dt <= Config.expansionLookbackMax
        }

        let older: HistorySample?
        if let preferred = olderCandidates.min(by: {
            abs($0.timestamp - (newest.timestamp - 0.9)) < abs($1.timestamp - (newest.timestamp - 0.9))
        }) {
            older = preferred
        } else {
            older = trackHistory.first.flatMap { first in
                let dt = newest.timestamp - first.timestamp
                return dt >= Config.expansionLookbackMin ? first : nil
            }
        }

        guard let old = older else { return nil }
        let deltaTime = newest.timestamp - old.timestamp
        guard deltaTime > 1e-3, old.smoothedScale > 1e-6 else { return nil }
        let rate = (newest.smoothedScale - old.smoothedScale) / old.smoothedScale / deltaTime
        guard rate.isFinite else { return nil }
        return rate
    }

    private enum ProposedLevel {
        case clear
        case ahead
        case close
    }

    private func evaluateRiskLevel(expansion: Double?, timestamp: TimeInterval) -> ProposedLevel {
        guard let lead = currentLead, let started = trackStartedAt else {
            return .clear
        }

        let age = timestamp - started
        let area = boxArea(lead.boundingBox)
        let inCorridor = currentlyInCorridor

        // Close requires corridor + age + (large area OR strong expansion OR persistent near).
        // Persistence streak is applied separately — never promote to close from one frame alone.
        if inCorridor, age >= Config.minimumCloseTrackAge {
            let large = area >= Config.closeMinArea
            let expanding = (expansion ?? 0) >= Config.closeMinExpansionPerSec
            let recent = trackHistory.suffix(3)
            let persistentNear = recent.count >= 3
                && recent.allSatisfy(\.inCorridor)
                && area >= Config.closeMinArea * 0.70
            if large || expanding || persistentNear {
                return .close
            }
            return .ahead
        }

        if age >= 0.15 || consecutiveHits >= Config.aheadPersistCount {
            return .ahead
        }
        return .clear
    }

    private func applyPersistenceAndHysteresis(proposed: ProposedLevel) {
        let previous = state

        switch proposed {
        case .close:
            closeStreak += 1
            aheadStreak += 1
            saferStreak = 0
        case .ahead:
            aheadStreak += 1
            closeStreak = 0
            saferStreak = 0
        case .clear:
            closeStreak = 0
            aheadStreak = 0
            saferStreak += 1
        }

        var next = previous

        switch previous {
        case .clear:
            if proposed == .close, closeStreak >= Config.closePersistCount {
                next = .close
            } else if (proposed == .ahead || proposed == .close),
                      aheadStreak >= Config.aheadPersistCount {
                next = .ahead
            }

        case .ahead:
            if proposed == .close, closeStreak >= Config.closePersistCount {
                next = .close
            } else if proposed == .clear, saferStreak >= Config.aheadDowngradeSafeCount {
                next = .clear
            }

        case .close:
            if proposed == .close {
                next = .close
            } else if saferStreak >= Config.closeDowngradeSafeCount {
                if proposed == .ahead, aheadStreak >= Config.aheadPersistCount {
                    next = .ahead
                } else if currentLead == nil {
                    next = .clear
                } else {
                    next = .ahead
                }
            }
        }

        if next != previous {
            transitionLog(from: previous, to: next)
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

        leadPerson = currentLead
        lastResult = PedestrianRiskResult(
            state: state,
            leadPerson: currentLead,
            relativeExpansionRate: expansion,
            trackAge: age,
            isInCorridor: currentlyInCorridor
        )
    }

    private func publishClear(timestamp: TimeInterval) {
        let previous = state
        clearTrack(reason: nil)
        if previous != .clear {
            transitionLog(from: previous, to: .clear)
        }
        state = .clear
        leadPerson = nil
        lastResult = .empty
        _ = timestamp
    }

    private func maybeLogDiagnostic(expansion: Double?) {
        #if DEBUG
        guard let lead = currentLead, updateCount % Config.diagnosticEveryNUpdates == 0 else { return }
        let area = boxArea(lead.boundingBox)
        let bottom = bottomCenter(of: lead.boundingBox)
        let expText = expansion.map { String(format: "%.2f/s", $0) } ?? "n/a"
        let stateLabel: String
        switch state {
        case .clear: stateLabel = "clear"
        case .ahead: stateLabel = "ahead"
        case .close: stateLabel = "close"
        }
        print(
            String(
                format: "[Pedestrian] conf=%.2f area=%.3f bottom=(%.2f,%.2f) corridor=%@ expansion=%@ state=%@",
                lead.confidence,
                Double(area),
                Double(bottom.x),
                Double(bottom.y),
                currentlyInCorridor ? "true" : "false",
                expText,
                stateLabel
            )
        )
        #endif
    }

    private func transitionLog(from: PedestrianRiskState, to: PedestrianRiskState) {
        #if DEBUG
        print("[PedestrianRisk] \(label(from)) -> \(label(to))")
        #endif
    }

    private func label(_ state: PedestrianRiskState) -> String {
        switch state {
        case .clear: return "CLEAR"
        case .ahead: return "AHEAD"
        case .close: return "CLOSE"
        }
    }

    // MARK: - Geometry

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
}
