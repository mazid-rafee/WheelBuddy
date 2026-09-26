//
//  GeometricLaneBackend.swift
//  WheelBuddy
//
//  Frame-level geometric lane perception (Vision contours). No temporal / speed / drift logic.
//

import CoreGraphics
import CoreImage
import CoreVideo
import Foundation
import Vision

protocol LanePerceptionBackend: AnyObject, Sendable {
    func analyze(pixelBuffer: CVPixelBuffer) -> LanePerceptionResult
}

/// Current production perception path: Core Image + `VNDetectContoursRequest` on upright portrait rear frames.
final class GeometricLaneBackend: LanePerceptionBackend, Sendable {
    /// Vision orientation — MultiCam rotates rear 90° for portrait, no mirror → upright buffers.
    static let visionOrientation: CGImagePropertyOrientation = .up

    // Trapezoidal ROI in top-left normalized image coords (y=0 top, y=1 bottom).
    static let roiTopY: CGFloat = 0.45
    static let roiBottomY: CGFloat = 0.95
    static let roiTopLeftX: CGFloat = 0.35
    static let roiTopRightX: CGFloat = 0.65
    static let roiBottomLeftX: CGFloat = 0.05
    static let roiBottomRightX: CGFloat = 0.95

    /// Near-field row used to estimate lane width / center.
    static let evaluationY: CGFloat = 0.85

    static let minLaneWidth: CGFloat = 0.18
    static let maxLaneWidth: CGFloat = 0.85
    static let minSegmentLength: CGFloat = 0.06
    static let maxAbsSlopeForHorizontalReject: CGFloat = 0.12 // dy/dx ~ flat

    func analyze(pixelBuffer: CVPixelBuffer) -> LanePerceptionResult {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        return analyze(ciImage: ciImage)
    }

    private func analyze(ciImage: CIImage) -> LanePerceptionResult {
        let mono = ciImage.applyingFilter("CIPhotoEffectMono", parameters: [:])
        let contrast = mono.applyingFilter("CIColorControls", parameters: [
            kCIInputContrastKey: 1.45,
            kCIInputBrightnessKey: 0.02,
            kCIInputSaturationKey: 0.0
        ])

        let request = VNDetectContoursRequest()
        request.contrastAdjustment = 1.0
        request.detectsDarkOnLight = false // light lane paint on darker asphalt
        request.maximumImageDimension = 512

        let handler = VNImageRequestHandler(
            ciImage: contrast,
            orientation: Self.visionOrientation,
            options: [:]
        )

        do {
            try handler.perform([request])
        } catch {
            return .empty
        }

        guard let observation = request.results?.first as? VNContoursObservation else {
            return .empty
        }

        var leftCandidates: [LaneCandidate] = []
        var rightCandidates: [LaneCandidate] = []

        let contourCount = observation.contourCount
        for index in 0..<contourCount {
            guard let contour = try? observation.contour(at: index) else { continue }
            let points = contour.normalizedPoints
            guard points.count >= 4 else { continue }

            var xs: [CGFloat] = []
            var ys: [CGFloat] = []
            xs.reserveCapacity(points.count)
            ys.reserveCapacity(points.count)

            for i in 0..<points.count {
                // Vision normalized: origin bottom-left → convert to top-left to match ROI constants.
                let vx = CGFloat(points[i].x)
                let vyBottomOrigin = CGFloat(points[i].y)
                let yTopOrigin = 1.0 - vyBottomOrigin
                guard Self.pointInROI(x: vx, y: yTopOrigin) else { continue }
                xs.append(vx)
                ys.append(yTopOrigin)
            }

            guard xs.count >= 4 else { continue }

            guard let xFit = Self.linearFitXGivenY(xs: xs, ys: ys) else { continue }

            let y0 = ys.min()!
            let y1 = ys.max()!
            let length = abs(y1 - y0)
            guard length >= Self.minSegmentLength else { continue }

            let x0 = xFit.a * y0 + xFit.b
            let x1 = xFit.a * y1 + xFit.b
            let dx = x1 - x0
            let dy = y1 - y0
            // Reject nearly horizontal in image space.
            if abs(dy) < 1e-4 { continue }
            if abs(dx) > 1e-4 && abs(dy / dx) < Self.maxAbsSlopeForHorizontalReject {
                continue
            }

            let xAtEval = xFit.a * Self.evaluationY + xFit.b
            guard xAtEval > 0.02, xAtEval < 0.98 else { continue }

            let midX = (x0 + x1) * 0.5
            // Inward toward vanishing (smaller y): left a≤0, right a≥0 in x = a*y + b (y down).
            let inward: Bool
            if midX < 0.50 {
                inward = xFit.a < 0.05
            } else {
                inward = xFit.a > -0.05
            }

            let score = length * (inward ? 1.4 : 0.6) * CGFloat(min(xs.count, 40)) / 40.0
            let candidate = LaneCandidate(
                xAtEval: xAtEval,
                score: score,
                length: length,
                a: xFit.a,
                b: xFit.b,
                y0: y0,
                y1: y1
            )

            if midX < 0.50, xAtEval < 0.55 {
                leftCandidates.append(candidate)
            } else if midX >= 0.50, xAtEval > 0.45 {
                rightCandidates.append(candidate)
            }
        }

        let bestLeft = leftCandidates.max(by: { $0.score < $1.score })
        let bestRight = rightCandidates.max(by: { $0.score < $1.score })

        var confidence: CGFloat = 0
        if let l = bestLeft, let r = bestRight, r.xAtEval > l.xAtEval {
            let width = r.xAtEval - l.xAtEval
            let widthOK = width >= Self.minLaneWidth && width <= Self.maxLaneWidth
            let lengthScore = min(1.0, (l.length + r.length) / 0.35)
            let widthScore: CGFloat = widthOK ? 1.0 : 0.25
            confidence = 0.35 * lengthScore + 0.45 * widthScore + 0.20 * min(1.0, (l.score + r.score) / 8.0)
            if !widthOK { confidence *= 0.5 }
            return LanePerceptionResult(
                leftLanePoints: Self.sampleLine(candidate: l),
                rightLanePoints: Self.sampleLine(candidate: r),
                leftXAtEvaluationY: l.xAtEval,
                rightXAtEvaluationY: r.xAtEval,
                confidence: min(1.0, confidence)
            )
        }

        if bestLeft != nil || bestRight != nil {
            confidence = 0.22
        }
        return LanePerceptionResult(
            leftLanePoints: bestLeft.map { Self.sampleLine(candidate: $0) } ?? [],
            rightLanePoints: bestRight.map { Self.sampleLine(candidate: $0) } ?? [],
            leftXAtEvaluationY: bestLeft?.xAtEval,
            rightXAtEvaluationY: bestRight?.xAtEval,
            confidence: confidence
        )
    }

    // MARK: - Geometry helpers

    private struct LaneCandidate {
        var xAtEval: CGFloat
        var score: CGFloat
        var length: CGFloat
        var a: CGFloat
        var b: CGFloat
        var y0: CGFloat
        var y1: CGFloat
    }

    private static func sampleLine(candidate: LaneCandidate, count: Int = 6) -> [CGPoint] {
        let lo = min(candidate.y0, candidate.y1)
        let hi = max(candidate.y0, candidate.y1)
        let steps = max(count, 2)
        return (0..<steps).map { i in
            let t = CGFloat(i) / CGFloat(steps - 1)
            let y = lo + (hi - lo) * t
            let x = candidate.a * y + candidate.b
            return CGPoint(x: min(1, max(0, x)), y: min(1, max(0, y)))
        }
    }

    private static func pointInROI(x: CGFloat, y: CGFloat) -> Bool {
        guard y >= roiTopY, y <= roiBottomY else { return false }
        let t = (y - roiTopY) / max(roiBottomY - roiTopY, 1e-6)
        let left = roiTopLeftX + (roiBottomLeftX - roiTopLeftX) * t
        let right = roiTopRightX + (roiBottomRightX - roiTopRightX) * t
        return x >= left && x <= right
    }

    private static func linearFitXGivenY(xs: [CGFloat], ys: [CGFloat]) -> (a: CGFloat, b: CGFloat)? {
        // x = a*y + b
        let n = CGFloat(ys.count)
        guard n >= 2 else { return nil }
        let sumY = ys.reduce(0, +)
        let sumX = xs.reduce(0, +)
        let sumYY = ys.reduce(0) { $0 + $1 * $1 }
        let sumYX = zip(ys, xs).reduce(CGFloat(0)) { $0 + $1.0 * $1.1 }
        let denom = n * sumYY - sumY * sumY
        guard abs(denom) > 1e-8 else { return nil }
        let a = (n * sumYX - sumY * sumX) / denom
        let b = (sumX - a * sumY) / n
        return (a, b)
    }
}
