//
//  CoreMLLaneBackend.swift
//  WheelBuddy
//
//  UFLDv2 CurveLanes ResNet18 lane perception.
//  Frame-level perception only — no EMA / speed / drift logic.
//

import CoreGraphics
import CoreImage
import CoreML
import CoreVideo
import Foundation

final class CoreMLLaneBackend: LanePerceptionBackend, @unchecked Sendable {

    // MARK: - Model

    private static let modelName = "UFLDv2_CurveLanes_ResNet18"

    private static let modelWidth = 1600
    private static let modelHeight = 800

    // CurveLanes UFLDv2 configuration.
    private static let numGridRow = 200
    private static let numRows = 72
    private static let numGridCol = 100
    private static let numCols = 41
    private static let numLanes = 10

    // Wheel Buddy uses this same near-field row.
    private static let evaluationY: CGFloat = 0.85
    private static let vehicleCenterX: CGFloat = 0.50

    // Preserve the existing geometric lane-width sanity range.
    private static let minLaneWidth: CGFloat = 0.18
    private static let maxLaneWidth: CGFloat = 0.85

    // Official CurveLanes support thresholds:
    // row count > numRows / 4, column count > numCols / 4.
    private static let minimumRowPoints = 19
    private static let minimumColumnPoints = 11

    // Reject very weak candidate lanes before ego-pair selection.
    private static let minimumLaneConfidence: CGFloat = 0.55

    // Same crop position that worked best in our offline experiment.
    private static let cropBottomNormalized: CGFloat = 0.88

    // CurveLanes original geometry / preprocessing.
    private static let curveLanesAspect: CGFloat = 2560.0 / 1440.0
    private static let cropRatio: CGFloat = 0.80

    private let fallback: any LanePerceptionBackend

    private let ciContext = CIContext(options: [
        .cacheIntermediates: false
    ])

    private lazy var model: MLModel? = {
        let bundle = Bundle.main

        let url =
            bundle.url(
                forResource: Self.modelName,
                withExtension: "mlmodelc"
            )
            ??
            bundle.url(
                forResource: Self.modelName,
                withExtension: "mlmodelc",
                subdirectory: "ML"
            )

        guard let url else {
            print(
                "[Lane-CoreML] model resource not found; using geometric fallback"
            )
            return nil
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all

        do {
            let loaded = try MLModel(
                contentsOf: url,
                configuration: configuration
            )

            print(
                "[Lane-CoreML] UFLDv2 CurveLanes model loaded"
            )

            return loaded
        } catch {
            print(
                "[Lane-CoreML] model load failed: \(error)"
            )
            return nil
        }
    }()

    init(
        fallback: any LanePerceptionBackend = GeometricLaneBackend()
    ) {
        self.fallback = fallback
    }

    // MARK: - Public backend API

    func analyze(
        pixelBuffer: CVPixelBuffer
    ) -> LanePerceptionResult {

        guard let model else {
            return fallback.analyze(
                pixelBuffer: pixelBuffer
            )
        }

        guard let prepared = prepareModelInput(
            from: pixelBuffer
        ) else {
            return fallback.analyze(
                pixelBuffer: pixelBuffer
            )
        }

        do {
            let inputProvider = try MLDictionaryFeatureProvider(
                dictionary: [
                    "image": MLFeatureValue(
                        pixelBuffer: prepared.pixelBuffer
                    )
                ]
            )

            let output = try model.prediction(
                from: inputProvider
            )

            guard
                let locRow = output
                    .featureValue(for: "loc_row")?
                    .multiArrayValue,

                let existRow = output
                    .featureValue(for: "exist_row")?
                    .multiArrayValue,

                let locCol = output
                    .featureValue(for: "loc_col")?
                    .multiArrayValue,

                let existCol = output
                    .featureValue(for: "exist_col")?
                    .multiArrayValue
            else {
                return fallback.analyze(
                    pixelBuffer: pixelBuffer
                )
            }

            guard
                let locRowTensor = Tensor4(
                    locRow,
                    expectedShape: [
                        1,
                        Self.numGridRow,
                        Self.numRows,
                        Self.numLanes
                    ]
                ),

                let existRowTensor = Tensor4(
                    existRow,
                    expectedShape: [
                        1,
                        2,
                        Self.numRows,
                        Self.numLanes
                    ]
                ),

                let locColTensor = Tensor4(
                    locCol,
                    expectedShape: [
                        1,
                        Self.numGridCol,
                        Self.numCols,
                        Self.numLanes
                    ]
                ),

                let existColTensor = Tensor4(
                    existCol,
                    expectedShape: [
                        1,
                        2,
                        Self.numCols,
                        Self.numLanes
                    ]
                )
            else {
                print(
                    "[Lane-CoreML] unexpected output shape"
                )

                return fallback.analyze(
                    pixelBuffer: pixelBuffer
                )
            }

            let lanes = decodeCurveLanes(
                locRow: locRowTensor,
                existRow: existRowTensor,
                locCol: locColTensor,
                existCol: existColTensor,
                cropTop: prepared.cropTopNormalized,
                cropBottom: prepared.cropBottomNormalized
            )

            guard let neuralResult = selectEgoLanePair(
                from: lanes
            ) else {

                #if DEBUG
                print("[Lane-CoreML] neural pair unavailable")
                #endif

                return .empty
            }

            return neuralResult

        } catch {
            print(
                "[Lane-CoreML] prediction failed: \(error)"
            )

            return fallback.analyze(
                pixelBuffer: pixelBuffer
            )
        }
    }

    // MARK: - Preprocessing

    private struct PreparedInput {
        let pixelBuffer: CVPixelBuffer

        /// Normalized TOP-LEFT coordinates in original portrait frame.
        let cropTopNormalized: CGFloat
        let cropBottomNormalized: CGFloat
    }

    private func prepareModelInput(
        from pixelBuffer: CVPixelBuffer
    ) -> PreparedInput? {

        let source = CIImage(
            cvPixelBuffer: pixelBuffer
        )

        let extent = source.extent

        let width = extent.width
        let height = extent.height

        guard width > 0, height > 0 else {
            return nil
        }

        /*
         Reproduce the effective geometry of the successful
         Python CurveLanes preprocessing:

             source road crop
             -> resize to 1600x1000
             -> retain bottom 800 px

         Rather than allocating an intermediate 1600x1000 image,
         calculate the equivalent source region directly.

         Source crop height:
             width / (2560 / 1440)

         Top 20% is discarded because crop_ratio = 0.8.
        */

        let sourceRoadHeight =
            width / Self.curveLanesAspect

        let sourceRoadHeightNormalized =
            sourceRoadHeight / height

        let effectiveHeightNormalized =
            sourceRoadHeightNormalized
            * Self.cropRatio

        let cropBottom =
            Self.cropBottomNormalized

        let cropTop = max(
            0,
            cropBottom
            - effectiveHeightNormalized
        )

        guard cropBottom > cropTop else {
            return nil
        }

        // CIImage uses bottom-left coordinates.
        let cropRect = CGRect(
            x: extent.minX,
            y: extent.minY
                + height * (1.0 - cropBottom),
            width: width,
            height: height
                * (cropBottom - cropTop)
        )

        let cropped = source
            .cropped(to: cropRect)

        // Move cropped image origin to (0, 0).
        let translated = cropped.transformed(
            by: CGAffineTransform(
                translationX: -cropRect.minX,
                y: -cropRect.minY
            )
        )

        let scaleX =
            CGFloat(Self.modelWidth)
            / cropRect.width

        let scaleY =
            CGFloat(Self.modelHeight)
            / cropRect.height

        let resized = translated.transformed(
            by: CGAffineTransform(
                scaleX: scaleX,
                y: scaleY
            )
        )

        var outputBuffer: CVPixelBuffer?

        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Self.modelWidth,
            Self.modelHeight,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &outputBuffer
        )

        guard
            status == kCVReturnSuccess,
            let outputBuffer
        else {
            return nil
        }

        ciContext.render(
            resized,
            to: outputBuffer,
            bounds: CGRect(
                x: 0,
                y: 0,
                width: Self.modelWidth,
                height: Self.modelHeight
            ),
            colorSpace: CGColorSpaceCreateDeviceRGB()
        )

        return PreparedInput(
            pixelBuffer: outputBuffer,
            cropTopNormalized: cropTop,
            cropBottomNormalized: cropBottom
        )
    }

    // MARK: - UFLDv2 decoding

    private struct LaneCandidate {
        let laneIndex: Int
        let points: [CGPoint]
        let confidence: CGFloat
        let xAtEvaluationY: CGFloat
    }

    private func decodeCurveLanes(
        locRow: Tensor4,
        existRow: Tensor4,
        locCol: Tensor4,
        existCol: Tensor4,
        cropTop: CGFloat,
        cropBottom: CGFloat
    ) -> [LaneCandidate] {

        var candidates: [LaneCandidate] = []

        for laneIndex in 0..<Self.numLanes {

            var rowPoints: [CGPoint] = []
            var rowConfidences: [CGFloat] = []

            var colPoints: [CGPoint] = []
            var colConfidences: [CGFloat] = []

            rowPoints.reserveCapacity(Self.numRows)
            rowConfidences.reserveCapacity(Self.numRows)
            colPoints.reserveCapacity(Self.numCols)
            colConfidences.reserveCapacity(Self.numCols)

            // -----------------------------------------------------
            // ROW HEAD
            // Fixed row anchor (Y), network predicts X.
            // -----------------------------------------------------

            for rowIndex in 0..<Self.numRows {

                let absentLogit = existRow.value(
                    0,
                    0,
                    rowIndex,
                    laneIndex
                )

                let presentLogit = existRow.value(
                    0,
                    1,
                    rowIndex,
                    laneIndex
                )

                // Official decoder uses argmax over the two
                // existence classes.
                guard presentLogit > absentLogit else {
                    continue
                }

                guard let refinedGrid = refinedCoordinate(
                    tensor: locRow,
                    gridCount: Self.numGridRow,
                    anchorIndex: rowIndex,
                    laneIndex: laneIndex
                ) else {
                    continue
                }

                let xModel =
                    refinedGrid
                    / CGFloat(Self.numGridRow - 1)

                // CurveLanes row anchors: linspace(0.4, 1.0, 72).
                let rowFraction =
                    CGFloat(rowIndex)
                    / CGFloat(Self.numRows - 1)

                let yModel =
                    0.40
                    + 0.60 * rowFraction

                let yOriginal =
                    cropTop
                    + yModel
                    * (cropBottom - cropTop)

                rowPoints.append(
                    CGPoint(
                        x: clamp01(xModel),
                        y: clamp01(yOriginal)
                    )
                )

                rowConfidences.append(
                    CGFloat(
                        sigmoid(
                            Double(
                                presentLogit
                                - absentLogit
                            )
                        )
                    )
                )
            }

            // -----------------------------------------------------
            // COLUMN HEAD
            // Fixed column anchor (X), network predicts Y.
            // -----------------------------------------------------

            for colIndex in 0..<Self.numCols {

                let absentLogit = existCol.value(
                    0,
                    0,
                    colIndex,
                    laneIndex
                )

                let presentLogit = existCol.value(
                    0,
                    1,
                    colIndex,
                    laneIndex
                )

                guard presentLogit > absentLogit else {
                    continue
                }

                guard let refinedGrid = refinedCoordinate(
                    tensor: locCol,
                    gridCount: Self.numGridCol,
                    anchorIndex: colIndex,
                    laneIndex: laneIndex
                ) else {
                    continue
                }

                // CurveLanes column anchors: linspace(0.0, 1.0, 41).
                let xModel =
                    CGFloat(colIndex)
                    / CGFloat(Self.numCols - 1)

                let yModel =
                    refinedGrid
                    / CGFloat(Self.numGridCol - 1)

                let yOriginal =
                    cropTop
                    + yModel
                    * (cropBottom - cropTop)

                colPoints.append(
                    CGPoint(
                        x: clamp01(xModel),
                        y: clamp01(yOriginal)
                    )
                )

                colConfidences.append(
                    CGFloat(
                        sigmoid(
                            Double(
                                presentLogit
                                - absentLogit
                            )
                        )
                    )
                )
            }

            // CurveLanes accepts a lane when either head has enough
            // support. This is the key difference from the previous
            // row-only decoder.
            let rowAccepted =
                rowPoints.count >= Self.minimumRowPoints

            let colAccepted =
                colPoints.count >= Self.minimumColumnPoints

            guard rowAccepted || colAccepted else {
                continue
            }

            var mergedPoints: [CGPoint] = []
            var mergedConfidences: [CGFloat] = []

            if rowAccepted {
                mergedPoints.append(contentsOf: rowPoints)
                mergedConfidences.append(contentsOf: rowConfidences)
            }

            if colAccepted {
                mergedPoints.append(contentsOf: colPoints)
                mergedConfidences.append(contentsOf: colConfidences)
            }

            guard
                !mergedPoints.isEmpty,
                !mergedConfidences.isEmpty
            else {
                continue
            }

            let meanConfidence =
                mergedConfidences.reduce(0, +)
                / CGFloat(mergedConfidences.count)

            guard
                meanConfidence >= Self.minimumLaneConfidence
            else {
                continue
            }

            let sortedPoints =
                mergedPoints.sorted {
                    $0.y < $1.y
                }

            let sortedRowPoints =
                rowPoints.sorted {
                    $0.y < $1.y
                }

            // Prefer the row head at evaluationY because row anchors
            // directly predict X at a known Y. If row support does not
            // cover evaluationY, use the combined row+column curve.
            let xAtEvaluationY =
                (
                    rowAccepted
                    ? interpolateX(
                        points: sortedRowPoints,
                        at: Self.evaluationY
                    )
                    : nil
                )
                ??
                estimateXFromNearbyPoints(
                    points: sortedPoints,
                    at: Self.evaluationY
                )

            guard let xAtEvaluationY else {
                continue
            }

            candidates.append(
                LaneCandidate(
                    laneIndex: laneIndex,
                    points: sortedPoints,
                    confidence: meanConfidence,
                    xAtEvaluationY: clamp01(xAtEvaluationY)
                )
            )
        }

        return candidates
    }

    /// UFLDv2 local softmax refinement around the grid argmax
    /// with local_width = 1.
    private func refinedCoordinate(
        tensor: Tensor4,
        gridCount: Int,
        anchorIndex: Int,
        laneIndex: Int
    ) -> CGFloat? {

        var bestGridIndex = 0
        var bestValue =
            -Float.greatestFiniteMagnitude

        for gridIndex in 0..<gridCount {

            let value = tensor.value(
                0,
                gridIndex,
                anchorIndex,
                laneIndex
            )

            if value > bestValue {
                bestValue = value
                bestGridIndex = gridIndex
            }
        }

        let lower = max(
            0,
            bestGridIndex - 1
        )

        let upper = min(
            gridCount - 1,
            bestGridIndex + 1
        )

        var localMaximum =
            -Float.greatestFiniteMagnitude

        for index in lower...upper {

            localMaximum = max(
                localMaximum,
                tensor.value(
                    0,
                    index,
                    anchorIndex,
                    laneIndex
                )
            )
        }

        var weightedIndex = 0.0
        var weightSum = 0.0

        for index in lower...upper {

            let logit = tensor.value(
                0,
                index,
                anchorIndex,
                laneIndex
            )

            let weight = exp(
                Double(
                    logit - localMaximum
                )
            )

            weightedIndex +=
                Double(index) * weight

            weightSum += weight
        }

        guard weightSum > 0 else {
            return nil
        }

        return CGFloat(
            weightedIndex / weightSum
            + 0.5
        )
    }

    /// Rescue evaluation for a curved lane when the row head does
    /// not directly bracket evaluationY. Fits x(y) locally using
    /// nearby row+column points.
    private func estimateXFromNearbyPoints(
        points: [CGPoint],
        at targetY: CGFloat
    ) -> CGFloat? {

        let nearby = points
            .filter {
                abs($0.y - targetY) <= 0.10
            }
            .sorted {
                abs($0.y - targetY)
                    < abs($1.y - targetY)
            }

        guard nearby.count >= 3 else {
            return nil
        }

        let samples = Array(
            nearby.prefix(8)
        )

        let count =
            CGFloat(samples.count)

        let meanY =
            samples.reduce(CGFloat.zero) {
                $0 + $1.y
            } / count

        let meanX =
            samples.reduce(CGFloat.zero) {
                $0 + $1.x
            } / count

        var numerator: CGFloat = 0
        var denominator: CGFloat = 0

        for point in samples {

            let dy =
                point.y - meanY

            numerator +=
                dy
                * (point.x - meanX)

            denominator +=
                dy * dy
        }

        if denominator < 1e-6 {
            return meanX
        }

        let slope =
            numerator / denominator

        let intercept =
            meanX
            - slope * meanY

        let estimated =
            slope * targetY
            + intercept

        guard
            estimated >= -0.10,
            estimated <= 1.10
        else {
            return nil
        }

        return clamp01(estimated)
    }

    private func clamp01(
        _ value: CGFloat
    ) -> CGFloat {

        min(
            1,
            max(
                0,
                value
            )
        )
    }

    // MARK: - Ego-lane selection

    private func selectEgoLanePair(
        from candidates: [LaneCandidate]
    ) -> LanePerceptionResult? {

        let leftCandidates = candidates
            .filter {
                $0.xAtEvaluationY
                    < Self.vehicleCenterX
            }
            .sorted {
                $0.xAtEvaluationY
                    > $1.xAtEvaluationY
            }

        let rightCandidates = candidates
            .filter {
                $0.xAtEvaluationY
                    > Self.vehicleCenterX
            }
            .sorted {
                $0.xAtEvaluationY
                    < $1.xAtEvaluationY
            }

        guard
            !leftCandidates.isEmpty,
            !rightCandidates.isEmpty
        else {
            return nil
        }

        var bestPair:
            (
                left: LaneCandidate,
                right: LaneCandidate,
                score: CGFloat
            )?

        /*
         Prefer the nearest plausible pair surrounding vehicle center.

         The width check prevents the two halves of a double line
         or another very-close pair from becoming the ego lane.
        */

        for left in leftCandidates {

            for right in rightCandidates {

                let width =
                    right.xAtEvaluationY
                    - left.xAtEvaluationY

                guard
                    width >= Self.minLaneWidth,
                    width <= Self.maxLaneWidth
                else {
                    continue
                }

                let minimumConfidence = min(
                    left.confidence,
                    right.confidence
                )

                /*
                 Smaller valid width is preferred because
                 the ego lane should be the nearest surrounding
                 pair, with confidence used as a tie-breaker.
                */

                let score =
                    width
                    - 0.10
                    * minimumConfidence

                if bestPair == nil
                    || score < bestPair!.score {

                    bestPair = (
                        left,
                        right,
                        score
                    )
                }
            }
        }

        guard let pair = bestPair else {
            return nil
        }

        let minimumConfidence = min(
            pair.left.confidence,
            pair.right.confidence
        )

        let minimumCoverage = min(
            CGFloat(pair.left.points.count)
                / CGFloat(Self.numRows),

            CGFloat(pair.right.points.count)
                / CGFloat(Self.numRows)
        )

        let normalizedCoverage = min(
            1.0,
            minimumCoverage / 0.50
        )

        /*
         Keep confidence semantics compatible with the
         existing LaneDetectionService 0.45 availability gate.
        */

        let frameConfidence = min(
            1.0,
            0.75 * minimumConfidence
                + 0.25 * normalizedCoverage
        )


        #if DEBUG
        print(
            String(
                format:
                    "[Lane-CoreML] pair L%d=%.3f R%d=%.3f width=%.3f conf=%.2f",
                pair.left.laneIndex,
                Double(pair.left.xAtEvaluationY),
                pair.right.laneIndex,
                Double(pair.right.xAtEvaluationY),
                Double(
                    pair.right.xAtEvaluationY
                    - pair.left.xAtEvaluationY
                ),
                Double(frameConfidence)
            )
        )
        #endif
        
        return LanePerceptionResult(
            leftLanePoints:
                pair.left.points,

            rightLanePoints:
                pair.right.points,

            leftXAtEvaluationY:
                pair.left.xAtEvaluationY,

            rightXAtEvaluationY:
                pair.right.xAtEvaluationY,

            confidence:
                frameConfidence
        )
    }

    // MARK: - Geometry

    private func interpolateX(
        points: [CGPoint],
        at targetY: CGFloat
    ) -> CGFloat? {

        guard points.count >= 2 else {
            return nil
        }

        for index in 0..<(points.count - 1) {

            let p0 = points[index]
            let p1 = points[index + 1]

            guard
                p0.y <= targetY,
                targetY <= p1.y
            else {
                continue
            }

            let dy =
                p1.y - p0.y

            guard
                abs(dy) > 1e-6,
                dy <= 0.06
            else {
                continue
            }

            let t =
                (targetY - p0.y)
                / dy

            return p0.x
                + t
                * (p1.x - p0.x)
        }

        // Small tolerance for a point very close to evaluationY.
        if let nearest = points.min(
            by: {
                abs($0.y - targetY)
                    < abs($1.y - targetY)
            }
        ),
           abs(nearest.y - targetY) <= 0.015 {

            return nearest.x
        }

        return nil
    }

    // MARK: - Helpers

    private func sigmoid(
        _ value: Double
    ) -> Double {

        if value >= 0 {
            return 1.0
                / (
                    1.0
                    + exp(-value)
                )
        }

        let e = exp(value)

        return e / (1.0 + e)
    }

    private struct Tensor4 {

        let array: MLMultiArray

        init?(
            _ array: MLMultiArray,
            expectedShape: [Int]
        ) {
            let actualShape =
                array.shape.map {
                    Int(
                        truncating: $0
                    )
                }

            guard
                actualShape
                    == expectedShape
            else {
                return nil
            }

            self.array = array
        }

        func value(
            _ i0: Int,
            _ i1: Int,
            _ i2: Int,
            _ i3: Int
        ) -> Float {

            array[
                [
                    NSNumber(value: i0),
                    NSNumber(value: i1),
                    NSNumber(value: i2),
                    NSNumber(value: i3)
                ]
            ].floatValue
        }
    }
}
