//
//  RouteRiskScorerTests.swift
//  WheelBuddyTests
//

import XCTest
@testable import WheelBuddy

@MainActor
final class RouteRiskScorerTests: XCTestCase {

    func testAdjustedCellRiskFormula() {
        XCTAssertEqual(
            RouteRiskScorer.adjustedCellRisk(current: 0.01, cellP90: 0.10),
            0.037,
            accuracy: 0.000_001
        )
        XCTAssertEqual(
            RouteRiskScorer.adjustedCellRisk(current: 0.10, cellP90: 0.10),
            0.10,
            accuracy: 0.000_001
        )
    }

    func testUniqueCellsCountedOnceForRouteRisk() {
        let response = RoutePredictionResponse(
            requestID: "test",
            modelVersion: "best.pt",
            routes: [
                makePrediction(
                    id: "route_0",
                    cells: [
                        makeCell(id: "a", current: 0.01, highHour: 0.10),
                        makeCell(id: "a", current: 0.01, highHour: 0.10),
                        makeCell(id: "b", current: 0.10, highHour: 0.10)
                    ]
                )
            ]
        )
        let routes = RouteRiskScorer.applyingPredictionScores(
            to: [makeRoute(index: 0)],
            response: response
        )
        // 0.037 + 0.10 = 0.137
        XCTAssertEqual(routes[0].safetyScore ?? 0, 0.137, accuracy: 0.000_001)
    }

    func testOldAPIFallbackUsesPredictionSummarySum() {
        let routes = RouteRiskScorer.applyingPredictionScores(
            to: [makeRoute(index: 0), makeRoute(index: 1), makeRoute(index: 2)],
            response: makeLegacyResponse(sums: [
                "route_0": 0.165814,
                "route_1": 0.249322,
                "route_2": 0.217158
            ])
        )
        let ranked = RouteRiskScorer.rankedSafestFirst(routes)
        XCTAssertEqual(ranked.map(\.id), ["route_0", "route_2", "route_1"])
        XCTAssertEqual(routes.first(where: { $0.id == "route_0" })?.safetyScore, 0.165814)
    }

    func testHighOOVMarksInsufficientGrayWithZeroScore() {
        let routes = RouteRiskScorer.applyingPredictionScores(
            to: [makeRoute(index: 0), makeRoute(index: 1)],
            response: RoutePredictionResponse(
                requestID: "test",
                modelVersion: "best.pt",
                routes: [
                    RoutePrediction(
                        routeID: "route_0",
                        cellCount: 100,
                        scoredCellCount: 2,
                        outOfVocabularyCount: 98,
                        activeHourBinStart: 12,
                        predictionSummary: RoutePredictionSummary(mean: 0.01, maximum: 0.02, sum: 0.02),
                        timeBinScores: [],
                        cells: []
                    ),
                    RoutePrediction(
                        routeID: "route_1",
                        cellCount: 50,
                        scoredCellCount: 48,
                        outOfVocabularyCount: 2,
                        activeHourBinStart: 12,
                        predictionSummary: RoutePredictionSummary(mean: 0.01, maximum: 0.02, sum: 0.48),
                        timeBinScores: [],
                        cells: []
                    )
                ]
            )
        )
        let byID = Dictionary(uniqueKeysWithValues: routes.map { ($0.id, $0) })
        XCTAssertTrue(byID["route_0"]!.hasInsufficientSafetyInfo)
        XCTAssertEqual(byID["route_0"]!.safetyScore, 0)
        XCTAssertNil(byID["route_0"]!.safetyTier)
        XCTAssertFalse(byID["route_1"]!.hasInsufficientSafetyInfo)
        XCTAssertEqual(byID["route_1"]!.safetyTier, .safest)
    }

    func testChartUsesAdjustedBinsAndActiveMatchesRouteRisk() {
        var route = makeRoute(index: 0)
        route.safetyScore = 0.2
        let prediction = RoutePrediction(
            routeID: "route_0",
            cellCount: 10,
            scoredCellCount: 8,
            outOfVocabularyCount: 2,
            activeHourBinStart: 12,
            predictionSummary: RoutePredictionSummary(mean: 0.05, maximum: 0.08, sum: 0.4),
            timeBinScores: Self.allAdjustedBins(activeAdjustedRisk: 0.25),
            cells: []
        )
        let model = RouteSafetyDetailsModel.build(route: route, prediction: prediction)
        XCTAssertTrue(model.hourlySafetyChartAvailable)
        XCTAssertEqual(model.safetyScoreByHourBin[12], 0.8, accuracy: 0.000_001)
        XCTAssertEqual(model.safetyScoreByHourBin[0], 0.85, accuracy: 0.000_001)
    }

    func testLegacyBinsHideHourlyChart() {
        var route = makeRoute(index: 0)
        route.safetyScore = 0.2
        let prediction = RoutePrediction(
            routeID: "route_0",
            cellCount: 10,
            scoredCellCount: 8,
            outOfVocabularyCount: 2,
            activeHourBinStart: 12,
            predictionSummary: RoutePredictionSummary(mean: 0.05, maximum: 0.08, sum: 0.4),
            timeBinScores: [
                TimeBinSafetyScore(
                    hourBinStart: 12,
                    severityWeightedSum: 0.99,
                    adjustedSeverityWeightedSum: nil,
                    maxSeverityWeightedRate: 0.2,
                    meanPersonRate: 0.01,
                    meanPropertyRate: 0.01,
                    meanSocietyRate: 0.01,
                    meanOtherRate: 0.01,
                    cellCount: 8
                )
            ],
            cells: []
        )
        let model = RouteSafetyDetailsModel.build(route: route, prediction: prediction)
        XCTAssertFalse(model.hourlySafetyChartAvailable)
        XCTAssertEqual(model.safetyScoreByHourBin[12], 0.8, accuracy: 0.000_001)
    }

    func testTiesResolvedByGoogleResponseOrder() {
        var a = makeRoute(index: 0)
        var b = makeRoute(index: 1)
        a.safetyScore = 0.2
        b.safetyScore = 0.2
        XCTAssertTrue(RouteRiskScorer.saferThan(a, b))
        XCTAssertFalse(RouteRiskScorer.saferThan(b, a))
    }

    // MARK: - Helpers

    private static func allAdjustedBins(activeAdjustedRisk: Double) -> [TimeBinSafetyScore] {
        RouteSafetyDetailsModel.hourBinStarts.map { bin in
            let risk = bin == 12 ? activeAdjustedRisk : 0.15
            return TimeBinSafetyScore(
                hourBinStart: bin,
                severityWeightedSum: risk + 0.5,
                adjustedSeverityWeightedSum: risk,
                maxSeverityWeightedRate: risk,
                meanPersonRate: 0.01,
                meanPropertyRate: 0.01,
                meanSocietyRate: 0.01,
                meanOtherRate: 0.01,
                cellCount: 8
            )
        }
    }

    private func makePrediction(id: String, cells: [CellPrediction]) -> RoutePrediction {
        RoutePrediction(
            routeID: id,
            cellCount: cells.count,
            scoredCellCount: cells.count,
            outOfVocabularyCount: 0,
            activeHourBinStart: 12,
            predictionSummary: RoutePredictionSummary(mean: 0, maximum: 0, sum: 0),
            timeBinScores: [],
            cells: cells
        )
    }

    private func makeCell(id: String, current: Double, highHour: Double) -> CellPrediction {
        CellPrediction(
            sequenceIndex: 0,
            h3Cell: id,
            entryTimeUTC: "2026-09-26T04:00:00Z",
            localHour: 12,
            dayOfWeek: "Friday",
            month: "September",
            cityName: "miami",
            severityWeightedRate: current,
            highHourSeverityWeightedRate: highHour,
            totalRate: current,
            personRate: current,
            propertyRate: 0,
            societyRate: 0,
            otherRate: 0,
            hourBinStart: 12
        )
    }

    private func makeLegacyResponse(sums: [String: Double]) -> RoutePredictionResponse {
        RoutePredictionResponse(
            requestID: "test-request",
            modelVersion: "best.pt",
            routes: sums.keys.sorted().map { routeID in
                let sum = sums[routeID]!
                return RoutePrediction(
                    routeID: routeID,
                    cellCount: 10,
                    scoredCellCount: 8,
                    outOfVocabularyCount: 2,
                    activeHourBinStart: 12,
                    predictionSummary: RoutePredictionSummary(
                        mean: sum / 8,
                        maximum: sum / 4,
                        sum: sum
                    ),
                    timeBinScores: [
                        TimeBinSafetyScore(
                            hourBinStart: 12,
                            severityWeightedSum: sum,
                            adjustedSeverityWeightedSum: nil,
                            maxSeverityWeightedRate: sum / 4,
                            meanPersonRate: 0.01,
                            meanPropertyRate: 0.01,
                            meanSocietyRate: 0.01,
                            meanOtherRate: 0.01,
                            cellCount: 8
                        )
                    ],
                    cells: []
                )
            }
        )
    }

    private func makeRoute(index: Int) -> ComputedRoute {
        ComputedRoute(
            id: "route_\(index)",
            encodedPolyline: "_p~iF~ps|U",
            durationText: "10 min",
            durationSeconds: 600,
            distanceMeters: 1000,
            sourcePlaceID: "src",
            destinationPlaceID: "dst",
            routeLabels: [],
            isDefault: index == 0,
            responseIndex: index,
            steps: [
                ComputedRouteStep(
                    encodedPolyline: "_p~iF~ps|U",
                    distanceMeters: 1000,
                    staticDurationSeconds: 600
                )
            ],
            isExtractionReady: true,
            safetyScore: nil,
            safetyTier: nil,
            hasInsufficientSafetyInfo: false
        )
    }
}
