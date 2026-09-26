//
//  NavigationSessionModel.swift
//  WheelBuddy
//

import Combine
import CoreLocation
import Foundation
import UIKit

/// Owns navigation mode, selected places, preview route state, and GO / End actions.
@MainActor
final class NavigationSessionModel: ObservableObject {
    @Published private(set) var mode: NavigationMode = .chooseLocation
    @Published var sourceText = ""
    @Published var destinationText = ""
    @Published private(set) var selectedSource: SelectedPlace?
    @Published private(set) var selectedDestination: SelectedPlace?

    @Published private(set) var routeState: RouteLoadingState = .idle
    @Published private(set) var previewRoutes: [ComputedRoute] = []
    /// Selected preview route used by the summary card and GO / Navigation SDK flow.
    @Published private(set) var previewRoute: ComputedRoute?
    /// Explicit selected route ID (matches `ComputedRoute.id` / extraction `route_id`).
    @Published private(set) var selectedRouteID: String?
    /// Shared client departure timestamp for the latest successful Routes API request.
    @Published private(set) var routeDepartureTime: Date?

    /// Capture-only CrimePredictor state (does not affect mock scores / UI selection).
    @Published private(set) var routePredictionState: RoutePredictionState = .idle
    @Published private(set) var activePredictionRequestID: String?
    @Published private(set) var lastRoutePredictionResponse: RoutePredictionResponse?
    @Published private(set) var lastRoutePredictionError: String?

    @Published private(set) var isStartingNavigation = false
    @Published var statusMessage: String?
    @Published private(set) var navigationStartRequestID: UUID?
    @Published private(set) var navigationEndRequestID: UUID?

    private let routesService = GoogleRoutesService()
    private let crimePredictionService = CrimePredictionService.shared
    private var routeTask: Task<Void, Never>?
    private var predictionTask: Task<Void, Never>?
    private var routeGeneration = 0
    private var predictionGeneration = 0
    private var startupTimeoutTask: Task<Void, Never>?
    /// Google routes held until CrimePredictor safety scores are applied.
    private var pendingUnscoredRoutes: [ComputedRoute] = []

    var hasValidSource: Bool {
        guard let selectedSource else { return false }
        return CLLocationCoordinate2DIsValid(selectedSource.coordinate)
    }

    var hasValidDestination: Bool {
        guard let selectedDestination else { return false }
        return CLLocationCoordinate2DIsValid(selectedDestination.coordinate)
    }

    var hasValidPair: Bool {
        hasValidSource && hasValidDestination
    }

    var isGOVisible: Bool {
        // Hide during active guidance and while startup is in progress.
        mode != .navigation && !isStartingNavigation
    }

    var isGOEnabled: Bool {
        mode == .showRoute && routeState.isReady && !isStartingNavigation
    }

    // MARK: - Selection updates

    func applySelection(_ place: SelectedPlace, to field: DirectionsSearchField) {
        endActiveGuidanceIfNeeded()
        switch field {
        case .source:
            selectedSource = place
            sourceText = place.primaryDisplayName
        case .destination:
            selectedDestination = place
            destinationText = place.primaryDisplayName
        }
        reconcileModeAfterSelectionChange()
    }

    func clearField(_ field: DirectionsSearchField) {
        endActiveGuidanceIfNeeded()
        switch field {
        case .source:
            selectedSource = nil
            sourceText = ""
        case .destination:
            selectedDestination = nil
            destinationText = ""
        }
        reconcileModeAfterSelectionChange()
    }

    func handleTypedText(_ text: String, for field: DirectionsSearchField) {
        switch field {
        case .source:
            sourceText = text
            if let selectedSource, text != selectedSource.primaryDisplayName {
                endActiveGuidanceIfNeeded()
                self.selectedSource = nil
                reconcileModeAfterSelectionChange()
            }
        case .destination:
            destinationText = text
            if let selectedDestination, text != selectedDestination.primaryDisplayName {
                endActiveGuidanceIfNeeded()
                self.selectedDestination = nil
                reconcileModeAfterSelectionChange()
            }
        }
    }

    func swapLocations() {
        endActiveGuidanceIfNeeded()
        let previousSource = selectedSource
        let previousDestination = selectedDestination
        let previousSourceText = sourceText
        let previousDestinationText = destinationText

        selectedSource = previousDestination
        selectedDestination = previousSource
        sourceText = previousDestinationText
        destinationText = previousSourceText
        reconcileModeAfterSelectionChange()
    }

    // MARK: - Mode / route

    private func reconcileModeAfterSelectionChange() {
        if mode == .navigation {
            return
        }

        if hasValidPair {
            mode = .showRoute
            statusMessage = nil
            requestPreviewRoute()
        } else {
            mode = .chooseLocation
            clearPreviewRoute()
        }
    }

    private func requestPreviewRoute() {
        guard let source = selectedSource, let destination = selectedDestination else {
            clearPreviewRoute()
            return
        }

        routeTask?.cancel()
        routeGeneration += 1
        let generation = routeGeneration
        routeState = .loading
        resetPreviewSelectionState()
        statusMessage = nil

        let origin = source.coordinate
        let dest = destination.coordinate
        let sourceID = source.placeID
        let destinationID = destination.placeID

        routeTask = Task { [weak self] in
            guard let self else { return }
            do {
                let batch = try await self.routesService.computeDrivingRoutes(
                    from: origin,
                    to: dest
                )
                guard !Task.isCancelled, generation == self.routeGeneration else { return }

                let unscored = batch.candidates.map { candidate in
                    ComputedRoute(
                        id: candidate.routeID,
                        encodedPolyline: candidate.encodedPolyline,
                        durationText: candidate.durationText,
                        durationSeconds: candidate.durationSeconds,
                        distanceMeters: candidate.distanceMeters,
                        sourcePlaceID: sourceID,
                        destinationPlaceID: destinationID,
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

                self.installPreviewRoutes(unscored, departureTime: batch.departureTime)
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled, generation == self.routeGeneration else { return }
                self.cancelRoutePrediction(resetState: true)
                self.previewRoutes = []
                self.previewRoute = nil
                self.selectedRouteID = nil
                self.routeDepartureTime = nil
                self.routeState = .failed(error.localizedDescription)
                self.statusMessage = error.localizedDescription
            }
        }
    }

    /// Clears selection and preview routes when a new Routes API request begins.
    func resetPreviewSelectionState() {
        cancelRoutePrediction(resetState: true)
        pendingUnscoredRoutes = []
        previewRoutes = []
        previewRoute = nil
        selectedRouteID = nil
        routeDepartureTime = nil
    }

    /// Installs Google routes privately, then requests CrimePredictor scores before UI ready.
    func installPreviewRoutes(_ routes: [ComputedRoute], departureTime: Date) {
        let ordered = routes.sorted { $0.responseIndex < $1.responseIndex }

        // Keep map/cards empty until safety tiers exist — avoids flashing all-yellow (nil → medium).
        pendingUnscoredRoutes = ordered
        previewRoutes = []
        previewRoute = nil
        selectedRouteID = nil
        routeDepartureTime = departureTime
        routeState = .loading

        #if DEBUG
        let payloads = RouteExtractionBuilder.buildPayloads(
            routes: ordered,
            departureTime: departureTime
        )
        RouteExtractionBuilder.logPayloads(payloads)
        #endif

        requestCrimePredictions(for: ordered, departureTime: departureTime)
    }

    /// Updates the selected preview route without refetching or rescoring.
    func selectRoute(id: String) {
        #if DEBUG
        print("[ROUTE_TAP_DEBUG] selectRoute(id: \(id)) currentSelected=\(selectedRouteID ?? "nil") routes=\(previewRoutes.map(\.id))")
        #endif
        guard previewRoutes.contains(where: { $0.id == id }) else {
            #if DEBUG
            print("[ROUTE_TAP_DEBUG] selectRoute ignored — id not in previewRoutes")
            #endif
            return
        }
        guard let route = previewRoutes.first(where: { $0.id == id }) else { return }
        let previous = selectedRouteID
        guard previous != id else {
            #if DEBUG
            print("[ROUTE_TAP_DEBUG] selectRoute ignored — already selected")
            #endif
            return
        }
        selectedRouteID = id
        previewRoute = route
        #if DEBUG
        print("[ROUTE_SELECTION] previous=\(previous ?? "nil") selected=\(id)")
        #endif
    }

    /// Alias kept for call sites that used the earlier name.
    func selectPreviewRoute(id: String) {
        selectRoute(id: id)
    }

    /// Backend-ready extraction payloads for the currently loaded preview routes.
    /// Includes every route regardless of which candidate is selected on the map.
    func routeExtractionPayloads() -> [RouteExtractionPayload] {
        guard let routeDepartureTime else { return [] }
        return RouteExtractionBuilder.buildPayloads(
            routes: previewRoutes,
            departureTime: routeDepartureTime
        )
    }

    func retryPreviewRoute() {
        guard hasValidPair, mode == .showRoute else { return }
        requestPreviewRoute()
    }

    private func clearPreviewRoute() {
        routeTask?.cancel()
        routeTask = nil
        routeGeneration += 1
        resetPreviewSelectionState()
        routeState = .idle
    }

    // MARK: - Crime prediction (capture-only)

    /// Sends all extraction-ready routes to the local CrimePredictor API.
    /// On success, replaces provisional selection with model-based safety ranking.
    private func requestCrimePredictions(for routes: [ComputedRoute], departureTime: Date) {
        cancelRoutePrediction(resetState: false)

        let payloads = RouteExtractionBuilder.buildPayloads(
            routes: routes,
            departureTime: departureTime
        )
        guard !payloads.isEmpty else {
            routePredictionState = .failed("No extraction-ready routes to score.")
            lastRoutePredictionError = "No extraction-ready routes to score."
            lastRoutePredictionResponse = nil
            activePredictionRequestID = nil
            // Publish unscored Google routes so the user is not stuck loading.
            if previewRoutes.isEmpty, !pendingUnscoredRoutes.isEmpty {
                let fallback = pendingUnscoredRoutes
                pendingUnscoredRoutes = []
                previewRoutes = fallback
                selectedRouteID = fallback.first?.id
                previewRoute = fallback.first
            }
            routeState = .ready
            return
        }

        predictionGeneration += 1
        let generation = predictionGeneration
        let requestID = UUID().uuidString
        let request = RoutePredictionRequest(requestID: requestID, routes: payloads)
        let expectedRouteIDs = Set(payloads.map(\.routeID))

        activePredictionRequestID = requestID
        lastRoutePredictionError = nil
        routePredictionState = .loading

        predictionTask = Task { [weak self] in
            guard let self else { return }
            do {
                let response = try await self.crimePredictionService.predictRoutes(request)
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    self.handleCrimePredictionSuccess(
                        response,
                        generation: generation,
                        expectedRequestID: requestID,
                        expectedRouteIDs: expectedRouteIDs
                    )
                }
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    self.handleCrimePredictionFailure(
                        error,
                        generation: generation,
                        expectedRequestID: requestID
                    )
                }
            }
        }
    }

    private func handleCrimePredictionSuccess(
        _ response: RoutePredictionResponse,
        generation: Int,
        expectedRequestID: String,
        expectedRouteIDs: Set<String>
    ) {
        guard generation == predictionGeneration else { return }
        guard activePredictionRequestID == expectedRequestID else { return }
        guard response.requestID == expectedRequestID else {
            let message = "Mismatched prediction request_id."
            lastRoutePredictionError = message
            routePredictionState = .failed(message)
            return
        }
        let returnedIDs = Set(response.routes.map(\.routeID))
        guard returnedIDs == expectedRouteIDs else {
            let message = "Prediction route IDs do not match the current route set."
            lastRoutePredictionError = message
            routePredictionState = .failed(message)
            return
        }

        lastRoutePredictionResponse = response
        lastRoutePredictionError = nil
        routePredictionState = .ready(response)

        let baseRoutes = previewRoutes.isEmpty ? pendingUnscoredRoutes : previewRoutes
        let scored = RouteRiskScorer.applyingPredictionScores(
            to: baseRoutes,
            response: response
        )
        let safestID = RouteRiskScorer.safestRouteID(in: scored)
        pendingUnscoredRoutes = []
        previewRoutes = scored
        if let safestID,
           let safest = scored.first(where: { $0.id == safestID }) {
            selectedRouteID = safestID
            previewRoute = safest
        } else {
            selectedRouteID = scored.first?.id
            previewRoute = scored.first
        }
        routeState = .ready

        #if DEBUG
        RoutePredictionDebugLogging.logSummary(response)
        RouteRiskScorer.logScores(scored, selectedRouteID: selectedRouteID)
        #endif
    }

    private func handleCrimePredictionFailure(
        _ error: Error,
        generation: Int,
        expectedRequestID: String
    ) {
        guard generation == predictionGeneration else { return }
        guard activePredictionRequestID == expectedRequestID else { return }

        let message = error.localizedDescription
        lastRoutePredictionError = message
        routePredictionState = .failed(message)

        // Still surface Google routes so navigation is usable; tiers stay nil (neutral UI).
        if previewRoutes.isEmpty, !pendingUnscoredRoutes.isEmpty {
            let fallback = pendingUnscoredRoutes
            pendingUnscoredRoutes = []
            previewRoutes = fallback
            selectedRouteID = fallback.first?.id
            previewRoute = fallback.first
        }
        routeState = .ready

        #if DEBUG
        RoutePredictionDebugLogging.logFailure(requestID: expectedRequestID, error: error)
        #endif
    }

    private func cancelRoutePrediction(resetState: Bool) {
        predictionTask?.cancel()
        predictionTask = nil
        predictionGeneration += 1
        activePredictionRequestID = nil
        if resetState {
            routePredictionState = .idle
            lastRoutePredictionResponse = nil
            lastRoutePredictionError = nil
        }
    }

    // MARK: - GO / navigation lifecycle

    func requestStartNavigation() {
        guard mode != .navigation else { return }

        guard hasValidPair else {
            statusMessage = "Choose a starting point and destination first."
            return
        }

        guard routeState.isReady else {
            if routeState.isLoading {
                statusMessage = "Wait for the route to finish loading."
            } else if case .failed = routeState {
                statusMessage = "Fix the route error before starting navigation."
            } else {
                statusMessage = "Choose a starting point and destination first."
            }
            return
        }

        guard !isStartingNavigation else { return }

        isStartingNavigation = true
        statusMessage = nil
        navigationStartRequestID = UUID()
        beginStartupTimeout()
    }

    private func beginStartupTimeout() {
        startupTimeoutTask?.cancel()
        startupTimeoutTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 25_000_000_000)
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, self.isStartingNavigation, self.mode != .navigation else { return }
                self.handleNavigationStartupFailed(
                    "Navigation is taking too long. Please try again."
                )
            }
        }
    }

    private func cancelStartupTimeout() {
        startupTimeoutTask?.cancel()
        startupTimeoutTask = nil
    }

    func handleTermsRejected() {
        cancelStartupTimeout()
        isStartingNavigation = false
        navigationStartRequestID = nil
        mode = .showRoute
        statusMessage = "Accept Google Navigation terms to start guidance."
    }

    func handleNavigationStartupFailed(_ message: String) {
        cancelStartupTimeout()
        isStartingNavigation = false
        navigationStartRequestID = nil
        mode = .showRoute
        statusMessage = message
    }

    func handleNavigationStartupSucceeded() {
        print("[NAV 4] Model changing to navigation mode")
        cancelStartupTimeout()
        isStartingNavigation = false
        navigationStartRequestID = nil
        mode = .navigation
        statusMessage = nil
        UIApplication.shared.isIdleTimerDisabled = true
    }

    func requestEndNavigation() {
        guard mode == .navigation || isStartingNavigation else { return }
        navigationEndRequestID = UUID()
    }

    func handleNavigationEnded() {
        cancelStartupTimeout()
        isStartingNavigation = false
        navigationStartRequestID = nil
        navigationEndRequestID = nil
        UIApplication.shared.isIdleTimerDisabled = false
        mode = .showRoute
        if hasValidPair {
            if previewRoute == nil || !routeState.isReady {
                requestPreviewRoute()
            }
        } else {
            mode = .chooseLocation
            clearPreviewRoute()
        }
    }

    func handleArrivedAtDestination() {
        requestEndNavigation()
    }

    private func endActiveGuidanceIfNeeded() {
        guard mode == .navigation || isStartingNavigation else { return }
        requestEndNavigation()
    }
}
