//
//  RouteExtractionTests.swift
//  WheelBuddyTests
//

import XCTest
@testable import WheelBuddy

final class RouteExtractionTests: XCTestCase {

    private let sampleJSON = """
    {
      "routes": [
        {
          "distanceMeters": 5000,
          "duration": "1250s",
          "polyline": { "encodedPolyline": "overall_poly_0" },
          "routeLabels": ["DEFAULT_ROUTE"],
          "legs": [
            {
              "distanceMeters": 3000,
              "duration": "800s",
              "steps": [
                {
                  "distanceMeters": 750,
                  "staticDuration": "65s",
                  "polyline": { "encodedPolyline": "step_0_0" }
                },
                {
                  "distanceMeters": 2250,
                  "staticDuration": "3.5s",
                  "polyline": { "encodedPolyline": "step_0_1" }
                }
              ]
            },
            {
              "distanceMeters": 2000,
              "duration": "450s",
              "steps": [
                {
                  "distanceMeters": 2000,
                  "staticDuration": "65s",
                  "polyline": { "encodedPolyline": "step_0_2" }
                }
              ]
            }
          ]
        },
        {
          "distanceMeters": 5200,
          "duration": "1300s",
          "polyline": { "encodedPolyline": "overall_poly_1" },
          "routeLabels": [],
          "legs": [
            {
              "distanceMeters": 5200,
              "duration": "1300s",
              "steps": [
                {
                  "distanceMeters": 1000,
                  "staticDuration": "65s",
                  "polyline": { "encodedPolyline": "step_1_0" }
                },
                {
                  "distanceMeters": 4200,
                  "staticDuration": "3.5s",
                  "polyline": { "encodedPolyline": "step_1_1" }
                }
              ]
            }
          ]
        }
      ]
    }
    """.data(using: .utf8)!

    func testDecodesAllRoutesWithDeterministicIDs() throws {
        let candidates = try GoogleRoutesService.decodeCandidates(from: sampleJSON)
        XCTAssertEqual(candidates.count, 2)
        let ordered = candidates.sorted { $0.responseIndex < $1.responseIndex }
        XCTAssertEqual(ordered.map(\.routeID), ["route_0", "route_1"])
        XCTAssertEqual(ordered.map(\.responseIndex), [0, 1])
    }

    func testFlattensMultipleLegsAndPreservesStepOrder() throws {
        let candidates = try GoogleRoutesService.decodeCandidates(from: sampleJSON)
        let route0 = try XCTUnwrap(candidates.first { $0.routeID == "route_0" })
        XCTAssertEqual(route0.steps.count, 3)
        XCTAssertEqual(
            route0.steps.map(\.encodedPolyline),
            ["step_0_0", "step_0_1", "step_0_2"]
        )
        XCTAssertEqual(route0.steps.map(\.distanceMeters), [750, 2250, 2000])
    }

    func testParsesOverallAndStepDurationsIncludingFractional() throws {
        let candidates = try GoogleRoutesService.decodeCandidates(from: sampleJSON)
        let route0 = try XCTUnwrap(candidates.first { $0.routeID == "route_0" })
        XCTAssertEqual(route0.durationSeconds, 1250)
        XCTAssertEqual(route0.steps[0].staticDurationSeconds, 65)
        XCTAssertEqual(route0.steps[1].staticDurationSeconds, 3.5, accuracy: 0.000_001)
        XCTAssertEqual(route0.steps[2].staticDurationSeconds, 65)

        let route1 = try XCTUnwrap(candidates.first { $0.routeID == "route_1" })
        XCTAssertEqual(route1.steps.map(\.staticDurationSeconds), [65, 3.5])
    }

    func testSharedDepartureTimestampAcrossPayloads() throws {
        let candidates = try GoogleRoutesService.decodeCandidates(from: sampleJSON)
        let routes = Self.makeComputedRoutes(from: candidates)
        let departure = Date(timeIntervalSince1970: 1_758_321_300) // 2025-09-19T... fixed
        let payloads = RouteExtractionBuilder.buildPayloads(
            routes: routes,
            departureTime: departure
        )

        XCTAssertEqual(payloads.count, 2)
        let expected = RouteDepartureTimeFormatting.iso8601UTCString(from: departure)
        XCTAssertTrue(expected.hasSuffix("Z"))
        XCTAssertEqual(Set(payloads.map(\.departureTimeUTC)), [expected])
        XCTAssertEqual(payloads.map(\.routeID), ["route_0", "route_1"])
        XCTAssertEqual(payloads[0].durationSeconds, 1250)
        XCTAssertEqual(payloads[0].steps.count, 3)
    }

    func testJSONUsesRequiredSnakeCaseKeys() throws {
        let candidates = try GoogleRoutesService.decodeCandidates(from: sampleJSON)
        let routes = Self.makeComputedRoutes(from: candidates)
        let departure = Date(timeIntervalSince1970: 1_758_321_300)
        let payloads = RouteExtractionBuilder.buildPayloads(
            routes: routes,
            departureTime: departure
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(payloads)
        let json = try XCTUnwrap(String(data: data, encoding: .utf8))

        XCTAssertTrue(json.contains("\"route_id\""))
        XCTAssertTrue(json.contains("\"departure_time_utc\""))
        XCTAssertTrue(json.contains("\"duration_seconds\""))
        XCTAssertTrue(json.contains("\"encoded_polyline\""))
        XCTAssertTrue(json.contains("\"distance_meters\""))
        XCTAssertTrue(json.contains("\"static_duration_seconds\""))
        XCTAssertFalse(json.contains("\"routeID\""))
        XCTAssertFalse(json.contains("\"departureTimeUTC\""))
    }

    func testMalformedDurationsFailSafely() {
        XCTAssertNil(GoogleDurationParser.parseSeconds(nil))
        XCTAssertNil(GoogleDurationParser.parseSeconds(""))
        XCTAssertNil(GoogleDurationParser.parseSeconds("s"))
        XCTAssertNil(GoogleDurationParser.parseSeconds("abc"))
        XCTAssertNil(GoogleDurationParser.parseSeconds("-65s"))
        XCTAssertNil(GoogleDurationParser.parseSeconds("65"))
        XCTAssertNil(GoogleDurationParser.parseSeconds("NaNs"))
        XCTAssertNil(GoogleDurationParser.parseSeconds("infs"))
        XCTAssertNil(GoogleDurationParser.parseSeconds("Infinitys"))
        XCTAssertEqual(GoogleDurationParser.parseSeconds("65s"), 65)
        XCTAssertEqual(GoogleDurationParser.parseSeconds("3.5s"), 3.5)
        XCTAssertEqual(GoogleDurationParser.parseSeconds("1250s"), 1250)
        XCTAssertEqual(GoogleDurationParser.parseSeconds(" 3.5s "), 3.5)
    }

    func testMalformedExtractionDoesNotDropMapRoute() throws {
        let malformedJSON = """
        {
          "routes": [
            {
              "distanceMeters": 1000,
              "duration": "not-a-duration",
              "polyline": { "encodedPolyline": "still_valid_for_map" },
              "routeLabels": ["DEFAULT_ROUTE"],
              "legs": [
                {
                  "steps": [
                    {
                      "distanceMeters": 1000,
                      "staticDuration": "bad",
                      "polyline": { "encodedPolyline": "step" }
                    }
                  ]
                }
              ]
            },
            {
              "distanceMeters": 2000,
              "duration": "1250s",
              "polyline": { "encodedPolyline": "good_route" },
              "routeLabels": [],
              "legs": [
                {
                  "steps": [
                    {
                      "distanceMeters": 2000,
                      "staticDuration": "65s",
                      "polyline": { "encodedPolyline": "good_step" }
                    }
                  ]
                }
              ]
            }
          ]
        }
        """.data(using: .utf8)!

        let candidates = try GoogleRoutesService.decodeCandidates(from: malformedJSON)
        XCTAssertEqual(candidates.count, 2, "Malformed extraction must not prevent map-route candidates")

        let bad = try XCTUnwrap(candidates.first { $0.routeID == "route_0" })
        XCTAssertEqual(bad.encodedPolyline, "still_valid_for_map")
        XCTAssertFalse(bad.isExtractionReady)

        let good = try XCTUnwrap(candidates.first { $0.routeID == "route_1" })
        XCTAssertTrue(good.isExtractionReady)

        let routes = Self.makeComputedRoutes(from: candidates)
        let payloads = RouteExtractionBuilder.buildPayloads(
            routes: routes,
            departureTime: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(payloads.map(\.routeID), ["route_1"])
    }

    func testPayloadPreservesGoogleResponseOrderEvenIfDefaultSortedFirst() throws {
        let candidates = try GoogleRoutesService.decodeCandidates(from: sampleJSON)
        // decodeCandidates sorts default first for UI; builder must restore Google order.
        XCTAssertEqual(candidates.first?.routeID, "route_0")
        let routes = Self.makeComputedRoutes(from: candidates).reversed()
        let payloads = RouteExtractionBuilder.buildPayloads(
            routes: Array(routes),
            departureTime: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(payloads.map(\.routeID), ["route_0", "route_1"])
    }

    // MARK: - Helpers

    private static func makeComputedRoutes(from candidates: [RouteCandidateResult]) -> [ComputedRoute] {
        candidates.map { candidate in
            ComputedRoute(
                id: candidate.routeID,
                encodedPolyline: candidate.encodedPolyline,
                durationText: candidate.durationText,
                durationSeconds: candidate.durationSeconds,
                distanceMeters: candidate.distanceMeters,
                sourcePlaceID: "src",
                destinationPlaceID: "dst",
                routeLabels: candidate.routeLabels,
                isDefault: candidate.isDefault,
                responseIndex: candidate.responseIndex,
                steps: candidate.steps.map {
                    ComputedRouteStep(
                        encodedPolyline: $0.encodedPolyline,
                        distanceMeters: $0.distanceMeters,
                        staticDurationSeconds: $0.staticDurationSeconds
                    )
                },
                isExtractionReady: candidate.isExtractionReady,
                safetyScore: nil,
                safetyTier: nil,
                hasInsufficientSafetyInfo: false
            )
        }
    }
}
