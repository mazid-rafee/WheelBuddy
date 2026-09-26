//
//  NavigationView.swift
//  WheelBuddy
//

import SwiftUI
import CoreLocation
import Foundation
import GoogleMaps
import GoogleNavigation
import UIKit

/// Temporary presentation colors for SHOW ROUTE polylines were replaced by
/// `RouteSafetyStyle` (green / amber / red safety tiers).

/// Compact guidance payload published from the Navigation SDK for custom SwiftUI chrome.
private struct NavigationDisplayInfo: Equatable {
    var remainingTimeText: String
    var remainingDistanceText: String
    var instructionText: String
    var roadName: String
    var maneuverImage: UIImage?

    static let empty = NavigationDisplayInfo(
        remainingTimeText: "",
        remainingDistanceText: "",
        instructionText: "",
        roadName: "",
        maneuverImage: nil
    )

    static func == (lhs: NavigationDisplayInfo, rhs: NavigationDisplayInfo) -> Bool {
        lhs.remainingTimeText == rhs.remainingTimeText
            && lhs.remainingDistanceText == rhs.remainingDistanceText
            && lhs.instructionText == rhs.instructionText
            && lhs.roadName == rhs.roadName
            && lhs.maneuverImage === rhs.maneuverImage
    }
}

/// Navigation region backed by Google Maps / Navigation SDK.
struct NavigationView: View {
    @Binding var speedingState: SpeedingState
    /// Raw Navigation SDK speeding fraction for DriveView OVER LIMIT (nil = unavailable).
    @Binding var percentageAboveLimit: CGFloat?

    @Environment(\.colorScheme) private var colorScheme

    @StateObject private var session = NavigationSessionModel()
    @StateObject private var placesService = PlacesAutocompleteService()
    @StateObject private var userLocation = UserLocationProvider()

    @State private var isFetchingPlaceDetails = false
    @State private var pendingCurrentLocationSelection = false
    @State private var navigationDisplayInfo = NavigationDisplayInfo.empty
    @State private var navigationCameraToggleRequestID: UUID?
    @State private var navigationMyLocationRequestID: UUID?
    @State private var safetyDetailsRouteID: String?

    @FocusState private var focusedField: DirectionsSearchField?

    private var showDropdown: Bool {
        focusedField != nil && session.mode != .navigation
    }

    private var showCurrentLocationOption: Bool {
        focusedField == .source
    }

    var body: some View {
        ZStack(alignment: .top) {
            GoogleMapView(
                source: session.selectedSource,
                destination: session.selectedDestination,
                previewRoutes: session.previewRoutes,
                previewRoute: session.previewRoute,
                mode: session.mode,
                colorScheme: colorScheme,
                navigationStartRequestID: session.navigationStartRequestID,
                navigationEndRequestID: session.navigationEndRequestID,
                navigationCameraToggleRequestID: navigationCameraToggleRequestID,
                navigationMyLocationRequestID: navigationMyLocationRequestID,
                onTermsRejected: {
                    resetSpeedingState()
                    session.handleTermsRejected()
                },
                onNavigationFailed: {
                    resetSpeedingState()
                    session.handleNavigationStartupFailed($0)
                },
                onNavigationStarted: { 
                    print("[NAV 3] SwiftUI received navigation started")
                    session.handleNavigationStartupSucceeded() 
                },
                onNavigationEnded: {
                    resetSpeedingState()
                    session.handleNavigationEnded()
                },
                onArrived: { session.handleArrivedAtDestination() },
                onNavigationInfoUpdated: { info in
                    navigationDisplayInfo = info
                },
                onNavigationInfoCleared: {
                    navigationDisplayInfo = .empty
                },
                onSpeedingStateChanged: { newState in
                    applySpeedingState(newState)
                },
                onSpeedingPercentageChanged: { percentage in
                    applySpeedingPercentage(percentage)
                }
            )
            .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
                    .allowsHitTesting(false)
            }
            .accessibilityLabel("Navigation map")
            .zIndex(0)

            if showDropdown {
                Color.clear
                    .contentShape(Rectangle())
                    .onTapGesture { dismissSearchUI() }
                    .zIndex(1)
            }

            if session.mode != .navigation {
                searchOverlay
                    // Intrinsic height only — do not cover the map with a full-screen hit block.
                    .frame(maxWidth: .infinity, alignment: .top)
                    .allowsHitTesting(true)
                    .zIndex(2)
            }

            if session.mode == .navigation {
                VStack(alignment: .trailing, spacing: 8) {
                    navigationManeuverBar
                        .frame(maxWidth: .infinity, alignment: .leading)

                    HStack(spacing: 0) {
                        Spacer(minLength: 0)
                            .allowsHitTesting(false)

                        VStack(spacing: 8) {
                            navigationCameraButton
                            myLocationButton
                        }
                    }
                }
                .padding(.top, 12)
                .padding(.horizontal, 12)
                .frame(maxWidth: .infinity, alignment: .top)
                .zIndex(10)
            }

            VStack {
                // Pass route taps through empty space to the map; keep controls hittable.
                Spacer(minLength: 0)
                    .allowsHitTesting(false)
                bottomControlsBar
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
            .zIndex(3)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .ignoresSafeArea(.keyboard, edges: .bottom)
        .overlay {
            if let safetyDetailsModel {
                RouteSafetyInformationDetailsOverlay(
                    model: safetyDetailsModel,
                    onDismiss: { safetyDetailsRouteID = nil }
                )
                .ignoresSafeArea()
            }
        }
        .onChange(of: focusedField) { _, newValue in
            guard session.mode != .navigation else { return }
            guard let newValue else {
                placesService.clearResults()
                return
            }
            placesService.resetSession()
            let query = newValue == .source ? session.sourceText : session.destinationText
            if newValue == .source, query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                placesService.clearResults()
            } else {
                placesService.scheduleSearch(query: query, biasCoordinate: userLocation.coordinate)
            }
        }
        .onChange(of: userLocation.coordinate?.latitude) { _, _ in
            guard pendingCurrentLocationSelection else { return }
            if userLocation.hasValidCoordinate {
                pendingCurrentLocationSelection = false
                applyCurrentLocationSelection()
            }
        }
        .onAppear {
            userLocation.refreshCachedLocation()
        }
    }

    // MARK: - Overlays

    private var searchOverlay: some View {
        VStack(spacing: 8) {
            DirectionsSearchCard(
                sourceText: Binding(
                    get: { session.sourceText },
                    set: { session.sourceText = $0 }
                ),
                destinationText: Binding(
                    get: { session.destinationText },
                    set: { session.destinationText = $0 }
                ),
                focusedField: $focusedField,
                onSourceTextChange: { text in
                    handleSearchTextChange(text, for: .source)
                },
                onDestinationTextChange: { text in
                    handleSearchTextChange(text, for: .destination)
                },
                onClearSource: {
                    session.clearField(.source)
                    placesService.clearResults()
                    focusedField = .source
                },
                onClearDestination: {
                    session.clearField(.destination)
                    placesService.clearResults()
                    focusedField = .destination
                },
                onSwap: {
                    session.swapLocations()
                    placesService.clearResults()
                }
            )

            if showDropdown {
                PlacesAutocompleteDropdown(
                    field: focusedField ?? .destination,
                    predictions: placesService.predictions,
                    isLoading: placesService.isLoading || isFetchingPlaceDetails,
                    errorMessage: placesService.errorMessage,
                    showCurrentLocationOption: showCurrentLocationOption,
                    currentLocationAvailable: userLocation.hasValidCoordinate,
                    locationDeniedMessage: userLocation.locationDeniedMessage,
                    onSelectPrediction: { item in
                        Task { await selectPrediction(item) }
                    },
                    onSelectCurrentLocation: selectCurrentLocation
                )
            }

            if session.mode == .showRoute {
                routeOptionsSection
            }

            if session.isStartingNavigation && session.mode != .navigation {
                HStack(spacing: 10) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Starting navigation…")
                        .font(.subheadline.weight(.medium))
                    Spacer(minLength: 0)
                }
                .foregroundStyle(.primary)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color(.systemBackground))
                        .shadow(color: .black.opacity(0.12), radius: 6, y: 2)
                }
            }

            if let status = session.statusMessage, session.mode != .navigation, !session.isStartingNavigation {
                Text(status)
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background {
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(Color(.systemBackground).opacity(0.95))
                    }
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 12)
    }

    /// Display-only ordering: safest → unsafest. Does not mutate `previewRoutes` / extraction order.
    private var safetySortedPreviewRoutes: [ComputedRoute] {
        RouteRiskScorer.rankedSafestFirst(session.previewRoutes)
    }

    private var routeLoadingStatusText: String {
        if case .loading = session.routePredictionState {
            return "Scoring routes…"
        }
        return "Finding route…"
    }

    private var safetyDetailsModel: RouteSafetyDetailsModel? {
        guard let routeID = safetyDetailsRouteID,
              let route = session.previewRoutes.first(where: { $0.id == routeID }) else {
            return nil
        }
        let prediction = session.lastRoutePredictionResponse?.routes.first(where: { $0.routeID == routeID })
        return RouteSafetyDetailsModel.build(route: route, prediction: prediction)
    }

    @ViewBuilder
    private var routeOptionsSection: some View {
        if session.routeState.isLoading {
            routeStatusChrome {
                HStack(spacing: 10) {
                    ProgressView()
                        .controlSize(.small)
                    Text(routeLoadingStatusText)
                        .font(.subheadline.weight(.medium))
                    Spacer(minLength: 0)
                }
            }
        } else if session.routeState.isReady, !safetySortedPreviewRoutes.isEmpty {
            HStack(spacing: 6) {
                ForEach(Array(safetySortedPreviewRoutes.enumerated()), id: \.element.id) { index, route in
                    RouteOptionCard(
                        displayRank: index + 1,
                        route: route,
                        isSelected: route.id == session.selectedRouteID,
                        onSelect: { session.selectRoute(id: route.id) },
                        onLongPress: { safetyDetailsRouteID = route.id }
                    )
                    .frame(maxWidth: .infinity)
                }
            }
        } else if let failure = session.routeState.failureMessage {
            routeStatusChrome {
                HStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text(failure)
                        .font(.footnote)
                        .lineLimit(2)
                    Spacer(minLength: 0)
                    Button("Retry") { session.retryPreviewRoute() }
                        .font(.footnote.weight(.semibold))
                }
            }
        }
    }

    private func routeStatusChrome<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        content()
            .foregroundStyle(.primary)
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color(.systemBackground))
                    .shadow(color: .black.opacity(0.12), radius: 6, y: 2)
            }
    }

    @ViewBuilder
    private var bottomControlsBar: some View {
        if session.mode == .navigation {
            HStack(spacing: 8) {
                endNavigationButton
                tripSummaryBar
                    .frame(maxWidth: .infinity)
            }
            .frame(maxWidth: .infinity)
            .padding(.leading, 12)
            .padding(.trailing, 12)
            .padding(.bottom, 12)
            .safeAreaPadding(.bottom, 0)
        } else if session.isGOVisible {
            HStack(alignment: .bottom) {
                goButton
                Spacer(minLength: 0)
                    .allowsHitTesting(false)
                myLocationButton
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 16)
            .safeAreaPadding(.bottom, 0)
        }
    }

    private var goButton: some View {
        Button {
            #if DEBUG
            print("[NAV_GO] tapped isGOEnabled=\(session.isGOEnabled) mode=\(session.mode) selected=\(session.selectedRouteID ?? "nil")")
            #endif
            if session.isGOEnabled {
                focusedField = nil
                placesService.clearResults()
                session.requestStartNavigation()
            } else if session.mode == .chooseLocation {
                session.statusMessage = "Choose a starting point and destination first."
            } else {
                session.requestStartNavigation()
            }
        } label: {
            Text("GO")
                .font(.system(size: 20, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 56, height: 56)
                .background(goButtonColor, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                .shadow(color: .black.opacity(0.25), radius: 6, y: 3)
                .opacity(session.isGOEnabled ? 1.0 : 0.45)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Start navigation")
        .accessibilityHint(
            session.isGOEnabled
                ? "Starts turn-by-turn guidance"
                : "Choose a starting point and destination first."
        )
    }

    /// Always uses the safest-route green (`#34A853`).
    private var goButtonColor: Color {
        Color(uiColor: RouteSafetyStyle.baseColor(for: .safest))
    }

    private var endNavigationButton: some View {
        Button {
            session.requestEndNavigation()
        } label: {
            Text("End")
                .font(.headline.weight(.bold))
                .foregroundStyle(.white)
                .padding(.horizontal, 18)
                .frame(height: 48)
                .background(Color.red, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                .shadow(color: .black.opacity(0.25), radius: 6, y: 3)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("End navigation")
    }

    private var tripSummaryBar: some View {
        HStack(spacing: 10) {
            Spacer(minLength: 0)

            Text(tripSummaryTimeText)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Color.green)
                .lineLimit(1)
                .minimumScaleFactor(0.75)

            Rectangle()
                .fill(Color.white.opacity(0.28))
                .frame(width: 1, height: 18)

            Text(tripSummaryDistanceText)
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(Color.white.opacity(0.92))
                .lineLimit(1)
                .minimumScaleFactor(0.75)

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 48)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.black.opacity(0.92))
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(tripSummaryTimeText) remaining, \(tripSummaryDistanceText) remaining")
    }

    private var navigationManeuverBar: some View {
        HStack(spacing: 10) {
            if let image = navigationDisplayInfo.maneuverImage {
                Image(uiImage: image)
                    .renderingMode(.template)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 28, height: 28)
                    .foregroundStyle(.white)
            } else {
                Image(systemName: "arrow.triangle.turn.up.right.diamond.fill")
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 28, height: 28)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(maneuverPrimaryText)
                    .font(.system(size: 16.5, weight: .semibold))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .minimumScaleFactor(0.75)

                if let secondary = maneuverSecondaryText {
                    Text(secondary)
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.75))
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .minimumScaleFactor(0.75)
                }
            }

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .frame(maxWidth: .infinity, minHeight: 58, maxHeight: 64, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(red: 0.07, green: 0.42, blue: 0.38))
                .shadow(color: .black.opacity(0.28), radius: 8, y: 3)
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(maneuverAccessibilityLabel)
    }

    private var navigationCameraButton: some View {
        Button {
            print("[MAP CONTROL] Recenter tapped")
            navigationCameraToggleRequestID = UUID()
        } label: {
            ZStack {
                Circle()
                    .fill(Color(white: 0.12).opacity(0.94))
                    .shadow(color: .black.opacity(0.28), radius: 6, y: 3)

                Image(systemName: "location.north.fill")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 20, height: 20)

                Circle()
                    .fill(Color.red)
                    .frame(width: 5, height: 5)
                    .offset(y: -11)
            }
            .frame(width: 48, height: 48)
            .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .frame(width: 48, height: 48)
        .contentShape(Circle())
        .allowsHitTesting(true)
        .accessibilityLabel("Change navigation camera")
        .accessibilityHint("Recenters the navigation camera on your route")
    }

    private var myLocationButton: some View {
        Button {
            print("[MAP CONTROL] My Location tapped")
            navigationMyLocationRequestID = UUID()
        } label: {
            Image(systemName: "scope")
                .font(.system(size: 22, weight: .semibold))
                .foregroundStyle(Color(white: 0.35))
                .frame(width: 44, height: 44)
                .background(Color.white)
                .clipShape(Circle())
                .shadow(color: .black.opacity(0.25), radius: 3, y: 2)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .frame(width: 44, height: 44)
        .contentShape(Circle())
        .allowsHitTesting(true)
        .accessibilityLabel("My location")
        .accessibilityHint("Centers the map on your current location")
    }

    private var tripSummaryTimeText: String {
        let text = navigationDisplayInfo.remainingTimeText.trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? "-- min" : text
    }

    private var tripSummaryDistanceText: String {
        let text = navigationDisplayInfo.remainingDistanceText.trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? "-- mi" : text
    }

    private var maneuverPrimaryText: String {
        let instruction = navigationDisplayInfo.instructionText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !instruction.isEmpty { return instruction }

        let road = navigationDisplayInfo.roadName.trimmingCharacters(in: .whitespacesAndNewlines)
        if !road.isEmpty { return road }

        return "Starting guidance…"
    }

    private var maneuverSecondaryText: String? {
        let instruction = navigationDisplayInfo.instructionText.trimmingCharacters(in: .whitespacesAndNewlines)
        let road = navigationDisplayInfo.roadName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !road.isEmpty else { return nil }
        guard !instruction.isEmpty else { return nil }
        // Only show road as secondary when it adds distinct information.
        guard !instruction.localizedCaseInsensitiveContains(road) else { return nil }
        return road
    }

    private var maneuverAccessibilityLabel: String {
        if let secondary = maneuverSecondaryText {
            return "\(maneuverPrimaryText), \(secondary)"
        }
        return maneuverPrimaryText
    }

    // MARK: - Places selection

    private func handleSearchTextChange(_ text: String, for field: DirectionsSearchField) {
        session.handleTypedText(text, for: field)

        // Only query Places while this field is actively focused. Avoids restarting
        // autocomplete when selection code programmatically updates the text.
        guard focusedField == field else { return }

        placesService.scheduleSearch(
            query: text,
            biasCoordinate: userLocation.coordinate
        )
    }

    private func selectPrediction(_ item: PlacePredictionItem) async {
        guard let field = focusedField else { return }
        isFetchingPlaceDetails = true
        defer { isFetchingPlaceDetails = false }

        do {
            let place = try await placesService.fetchSelectedPlace(placeID: item.placeID)
            session.applySelection(place, to: field)
            dismissSearchUI()
        } catch {
            placesService.presentError(error.localizedDescription)
        }
    }

    private func selectCurrentLocation() {
        userLocation.clearDeniedMessage()
        switch userLocation.authorizationStatus {
        case .authorizedAlways, .authorizedWhenInUse:
            if userLocation.hasValidCoordinate {
                applyCurrentLocationSelection()
            } else {
                pendingCurrentLocationSelection = true
                userLocation.requestWhenInUseIfNeeded()
            }
        case .notDetermined:
            pendingCurrentLocationSelection = true
            userLocation.requestWhenInUseIfNeeded()
        case .denied, .restricted:
            userLocation.requestWhenInUseIfNeeded()
        @unknown default:
            break
        }
    }

    private func applyCurrentLocationSelection() {
        guard let coordinate = userLocation.coordinate,
              CLLocationCoordinate2DIsValid(coordinate) else { return }
        session.applySelection(.currentLocation(coordinate: coordinate), to: .source)
        dismissSearchUI()
    }

    private func dismissSearchUI() {
        focusedField = nil
        placesService.clearResults()
    }

    private func applySpeedingState(_ newState: SpeedingState) {
        guard speedingState != newState else { return }
        speedingState = newState
    }

    private func applySpeedingPercentage(_ percentage: CGFloat?) {
        percentageAboveLimit = percentage
    }

    private func resetSpeedingState() {
        applySpeedingState(.unavailable)
        applySpeedingPercentage(nil)
    }
}

// MARK: - Google Map + Navigation SDK

private struct GoogleMapView: UIViewRepresentable {
    var source: SelectedPlace?
    var destination: SelectedPlace?
    var previewRoutes: [ComputedRoute]
    var previewRoute: ComputedRoute?
    var mode: NavigationMode
    var colorScheme: ColorScheme
    var navigationStartRequestID: UUID?
    var navigationEndRequestID: UUID?
    var navigationCameraToggleRequestID: UUID?
    var navigationMyLocationRequestID: UUID?

    var onTermsRejected: () -> Void
    var onNavigationFailed: (String) -> Void
    var onNavigationStarted: () -> Void
    var onNavigationEnded: () -> Void
    var onArrived: () -> Void
    var onNavigationInfoUpdated: (NavigationDisplayInfo) -> Void
    var onNavigationInfoCleared: () -> Void
    var onSpeedingStateChanged: (SpeedingState) -> Void
    var onSpeedingPercentageChanged: (CGFloat?) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> GMSMapView {
        let options = GMSMapViewOptions()
        if let coordinate = context.coordinator.lastKnownCoordinate {
            options.camera = GMSCameraPosition.camera(withTarget: coordinate, zoom: 16)
        } else {
            options.camera = GMSCameraPosition.camera(
                withLatitude: 25.7617,
                longitude: -80.1918,
                zoom: 13
            )
        }

        let mapView = GMSMapView(options: options)
        mapView.isMyLocationEnabled = true
        mapView.settings.myLocationButton = false
        mapView.settings.compassButton = false
        mapView.settings.isRecenterButtonEnabled = false
        mapView.settings.isNavigationReportIncidentButtonEnabled = false
        mapView.settings.zoomGestures = true
        mapView.settings.scrollGestures = true
        mapView.settings.rotateGestures = true
        mapView.settings.tiltGestures = true
        mapView.shouldDisplaySpeedLimit = false
        mapView.shouldDisplaySpeedometer = false
        // Bottom inset matches GO button row so the location button shares the same baseline.
        mapView.padding = UIEdgeInsets(top: 150, left: 72, bottom: 12, right: 12)

        context.coordinator.attach(to: mapView)
        context.coordinator.syncCallbacks(from: self)
        context.coordinator.applyInterfaceStyle(colorScheme, navigationEnabled: false)
        context.coordinator.updateMarkers(source: source, destination: destination, mode: mode, previewRoutes: previewRoutes)
        context.coordinator.updatePreviewPolylines(previewRoutes, selectedRoute: previewRoute, mode: mode)
        return mapView
    }

        func updateUIView(_ mapView: GMSMapView, context: Context) {
        let coordinator = context.coordinator
        coordinator.syncCallbacks(from: self)
        coordinator.source = source
        coordinator.destination = destination
        coordinator.previewRoutes = previewRoutes
        coordinator.previewRoute = previewRoute
        coordinator.mode = mode
        coordinator.colorScheme = colorScheme
        coordinator.applyInterfaceStyle(colorScheme, navigationEnabled: mapView.isNavigationEnabled)
        // Apply padding before route fitting so camera fit uses the final layout insets.
        coordinator.applyModePresentation(mode)
        coordinator.updateMarkers(source: source, destination: destination, mode: mode, previewRoutes: previewRoutes)
        coordinator.updatePreviewPolylines(previewRoutes, selectedRoute: previewRoute, mode: mode)
        coordinator.handleNavigationStartIfNeeded(navigationStartRequestID)
        coordinator.handleNavigationEndIfNeeded(navigationEndRequestID)
        coordinator.handleCameraToggleIfNeeded(navigationCameraToggleRequestID)
        coordinator.handleMyLocationIfNeeded(navigationMyLocationRequestID)
    }

    final class Coordinator: NSObject, CLLocationManagerDelegate, GMSNavigatorListener {
        var source: SelectedPlace?
        var destination: SelectedPlace?
        var previewRoutes: [ComputedRoute] = []
        var previewRoute: ComputedRoute?
        var mode: NavigationMode = .chooseLocation
        var colorScheme: ColorScheme = .dark

        var onTermsRejected: () -> Void = {}
        var onNavigationFailed: (String) -> Void = { _ in }
        var onNavigationStarted: () -> Void = {}
        var onNavigationEnded: () -> Void = {}
        var onArrived: () -> Void = {}
        var onNavigationInfoUpdated: (NavigationDisplayInfo) -> Void = { _ in }
        var onNavigationInfoCleared: () -> Void = {}
        var onSpeedingStateChanged: (SpeedingState) -> Void = { _ in }
        var onSpeedingPercentageChanged: (CGFloat?) -> Void = { _ in }

        private let locationManager = CLLocationManager()
        private weak var mapView: GMSMapView?
        private var didApplyLiveFix = false
        private var sourceMarker: GMSMarker?
        private var destinationMarker: GMSMarker?
        private var previewPolylines: [String: GMSPolyline] = [:]
        private var lastFittedPairKey: String?
        private var lastPolylineGeometryKey: String?
        private var lastSelectedPolylineID: String?
        private var lastSourcePlaceID: String?
        private var lastDestinationPlaceID: String?
        private var handledStartID: UUID?
        private var handledEndID: UUID?
        private var handledCameraToggleRequestID: UUID?
        private var handledMyLocationRequestID: UUID?
        private var isStarting = false
        private var didRegisterNavigatorListener = false
        private var lastPublishedDisplayInfo = NavigationDisplayInfo.empty
        private var lastPublishedSpeedingState: SpeedingState = .unavailable
        private var lastPublishedSpeedingPercentage: CGFloat?
        private let distanceFormatter: MeasurementFormatter = {
            let formatter = MeasurementFormatter()
            formatter.unitStyle = .short
            formatter.locale = .current
            formatter.numberFormatter.maximumFractionDigits = 1
            formatter.numberFormatter.minimumFractionDigits = 0
            return formatter
        }()

        var lastKnownCoordinate: CLLocationCoordinate2D? {
            locationManager.location?.coordinate
        }

        override init() {
            super.init()
            locationManager.delegate = self
            locationManager.desiredAccuracy = kCLLocationAccuracyBest
        }

        func syncCallbacks(from parent: GoogleMapView) {
            onTermsRejected = parent.onTermsRejected
            onNavigationFailed = parent.onNavigationFailed
            onNavigationStarted = parent.onNavigationStarted
            onNavigationEnded = parent.onNavigationEnded
            onArrived = parent.onArrived
            onNavigationInfoUpdated = parent.onNavigationInfoUpdated
            onNavigationInfoCleared = parent.onNavigationInfoCleared
            onSpeedingStateChanged = parent.onSpeedingStateChanged
            onSpeedingPercentageChanged = parent.onSpeedingPercentageChanged
        }

        func attach(to mapView: GMSMapView) {
            self.mapView = mapView
            requestLocationAccessIfNeeded()
        }

        func applyInterfaceStyle(_ colorScheme: ColorScheme, navigationEnabled: Bool) {
            guard let mapView else { return }
            // Follow the SwiftUI / app color scheme instead of Maps' light-mode default.
            mapView.overrideUserInterfaceStyle = (colorScheme == .dark) ? .dark : .light
            if navigationEnabled {
                mapView.lightingMode = (colorScheme == .dark) ? .lowLight : .normal
            }
        }

        func applyModePresentation(_ mode: NavigationMode) {
            guard let mapView else { return }
            mapView.isMyLocationEnabled = true
            // Always use the custom SwiftUI My Location control.
            mapView.settings.myLocationButton = false
            mapView.settings.compassButton = false
            mapView.settings.isRecenterButtonEnabled = false
            mapView.settings.isNavigationReportIncidentButtonEnabled = false

            if mode == .navigation {
                // Leave room above End + full-width trip summary so Google attribution stays visible.
                mapView.padding = UIEdgeInsets(top: 72, left: 12, bottom: 80, right: 12)
                mapView.settings.isNavigationHeaderEnabled = false
                mapView.settings.isNavigationFooterEnabled = false
                setPreviewPolylinesVisible(false)
                applySpeedLimitDisplay(enabled: mapView.isNavigationEnabled)
            } else {
                // Symmetric insets for search chrome + GO / My Location row.
                // Keep modest so route-preview camera fit is not double-padded into a regional zoom.
                mapView.padding = UIEdgeInsets(top: 12, left: 12, bottom: 12, right: 12)
                setPreviewPolylinesVisible(mode == .showRoute)
                applySpeedLimitDisplay(enabled: false)
            }

            #if DEBUG
            print("[Navigation UI] mode=\(mode), myLocationButton=\(mapView.settings.myLocationButton)")
            #endif
        }

        /// Keeps Google's native speed-limit / speedometer controls off.
        /// OVER LIMIT beside GPS speed comes from Navigation SDK speeding % in DriveView.
        private func applySpeedLimitDisplay(enabled: Bool) {
            guard let mapView else { return }
            mapView.shouldDisplaySpeedLimit = false
            mapView.shouldDisplaySpeedometer = false
            _ = enabled
        }

        private func setPreviewPolylinesVisible(_ visible: Bool) {
            guard let mapView else { return }
            for polyline in previewPolylines.values {
                polyline.map = visible ? mapView : nil
            }
        }

        private func clearPreviewPolylines() {
            for polyline in previewPolylines.values {
                polyline.map = nil
            }
            previewPolylines.removeAll()
            lastPolylineGeometryKey = nil
            lastSelectedPolylineID = nil
        }

        private func applyDisabledGoogleNavigationControls(on mapView: GMSMapView) {
            mapView.settings.isNavigationHeaderEnabled = false
            mapView.settings.isNavigationFooterEnabled = false
            mapView.settings.isNavigationReportIncidentButtonEnabled = false
            mapView.settings.isRecenterButtonEnabled = false
            mapView.settings.compassButton = false
            mapView.settings.myLocationButton = false
        }

        func updateMarkers(
            source: SelectedPlace?,
            destination: SelectedPlace?,
            mode: NavigationMode,
            previewRoutes: [ComputedRoute]
        ) {
            guard let mapView else { return }

            let sourceChanged = lastSourcePlaceID != source?.placeID
            let destinationChanged = lastDestinationPlaceID != destination?.placeID
            lastSourcePlaceID = source?.placeID
            lastDestinationPlaceID = destination?.placeID

            if let source {
                let marker = sourceMarker ?? GMSMarker()
                marker.position = source.coordinate
                marker.title = source.primaryDisplayName
                marker.snippet = source.formattedAddress
                marker.icon = GMSMarker.markerImage(with: .systemBlue)
                marker.map = mode == .navigation ? nil : mapView
                sourceMarker = marker
            } else {
                sourceMarker?.map = nil
                sourceMarker = nil
            }

            if let destination {
                let marker = destinationMarker ?? GMSMarker()
                marker.position = destination.coordinate
                marker.title = destination.primaryDisplayName
                marker.snippet = destination.formattedAddress
                marker.icon = GMSMarker.markerImage(with: .systemRed)
                marker.map = mode == .navigation ? nil : mapView
                destinationMarker = marker
            } else {
                destinationMarker?.map = nil
                destinationMarker = nil
            }

            if sourceChanged || destinationChanged {
                lastFittedPairKey = nil
            }

            // Only refit for endpoint changes. Route polyline updates own the SHOW ROUTE camera fit.
            if mode == .showRoute, source != nil, destination != nil, sourceChanged || destinationChanged {
                fitCameraToRouteOrMarkers(
                    source: source,
                    destination: destination,
                    routes: previewRoutes,
                    force: true
                )
            }
        }

        func updatePreviewPolylines(
            _ routes: [ComputedRoute],
            selectedRoute: ComputedRoute?,
            mode: NavigationMode
        ) {
            guard let mapView else { return }

            guard mode == .showRoute, !routes.isEmpty else {
                clearPreviewPolylines()
                return
            }

            let selectedID = selectedRoute?.id
                ?? routes.first(where: { $0.safetyTier == .safest })?.id
                ?? routes.first?.id
            // Include geometry fingerprint so a new Routes response with the same IDs still redraws.
            let geometryKey = routes
                .map { "\($0.id)#\($0.encodedPolyline.count):\($0.distanceMeters)" }
                .joined(separator: "||")

            // Same route geometries: restyle selection only — do not recreate or refit camera.
            if geometryKey == lastPolylineGeometryKey, !previewPolylines.isEmpty {
                if selectedID != lastSelectedPolylineID {
                    applySafetyStyles(to: routes, selectedID: selectedID)
                    lastSelectedPolylineID = selectedID
                } else {
                    // Keep userData associated for styling lookups across SwiftUI redraws.
                    for route in routes {
                        guard let polyline = previewPolylines[route.id] else { continue }
                        polyline.isTappable = false
                        polyline.userData = route.id
                    }
                }
                setPreviewPolylinesVisible(true)
                return
            }

            clearPreviewPolylines()
            lastPolylineGeometryKey = geometryKey
            lastSelectedPolylineID = selectedID

            // Draw unselected first, selected last so z-order is correct even before zIndex.
            let ordered = routes.sorted { lhs, rhs in
                let lhsSelected = lhs.id == selectedID
                let rhsSelected = rhs.id == selectedID
                if lhsSelected != rhsSelected {
                    return !lhsSelected && rhsSelected
                }
                return lhs.responseIndex < rhs.responseIndex
            }

            var didDrawAny = false
            for route in ordered {
                guard let path = GMSPath(fromEncodedPath: route.encodedPolyline), path.count() > 0 else {
                    continue
                }
                let isSelected = route.id == selectedID
                let tier = route.safetyTier
                let polyline = GMSPolyline(path: path)
                polyline.strokeWidth = RouteSafetyStyle.strokeWidth(isSelected: isSelected)
                polyline.strokeColor = RouteSafetyStyle.strokeColor(tier: tier, isSelected: isSelected)
                polyline.geodesic = true
                polyline.zIndex = RouteSafetyStyle.zIndex(isSelected: isSelected)
                polyline.isTappable = false
                polyline.userData = route.id
                polyline.map = mapView
                previewPolylines[route.id] = polyline
                didDrawAny = true
                #if DEBUG
                print(
                    "[Routes] created polyline id=\(route.id) tappable=\(polyline.isTappable) width=\(polyline.strokeWidth) z=\(polyline.zIndex)"
                )
                #endif
            }

            #if DEBUG
            print("[Routes] drew \(previewPolylines.count) polyline(s); selected=\(selectedID ?? "nil")")
            #endif

            if didDrawAny {
                lastFittedPairKey = nil
                fitCameraToRouteOrMarkers(
                    source: source,
                    destination: destination,
                    routes: routes,
                    force: true
                )
            } else if mode == .showRoute {
                fitCameraToRouteOrMarkers(
                    source: source,
                    destination: destination,
                    routes: [],
                    force: true
                )
            }
        }

        /// Updates stroke color/width/zIndex for existing polylines without recreating geometry.
        private func applySafetyStyles(to routes: [ComputedRoute], selectedID: String?) {
            for route in routes {
                guard let polyline = previewPolylines[route.id] else { continue }
                let isSelected = route.id == selectedID
                let tier = route.safetyTier
                polyline.strokeColor = RouteSafetyStyle.strokeColor(tier: tier, isSelected: isSelected)
                polyline.strokeWidth = RouteSafetyStyle.strokeWidth(isSelected: isSelected)
                polyline.zIndex = RouteSafetyStyle.zIndex(isSelected: isSelected)
                polyline.isTappable = false
                polyline.userData = route.id
            }
        }

        private func fitCameraToRouteOrMarkers(
            source: SelectedPlace?,
            destination: SelectedPlace?,
            routes: [ComputedRoute],
            force: Bool
        ) {
            guard mode == .showRoute, let mapView else { return }

            let insets = UIEdgeInsets(
                top: 170,
                left: 56,
                bottom: 88,
                right: 56
            )

            if !routes.isEmpty {
                let pairKey = "routes:" + routes.map(\.id).joined(separator: "|")
                if !force, pairKey == lastFittedPairKey { return }

                var combinedBounds: GMSCoordinateBounds?
                for route in routes {
                    guard let path = GMSPath(fromEncodedPath: route.encodedPolyline) else { continue }
                    for index in 0..<path.count() {
                        let coordinate = path.coordinate(at: index)
                        guard CLLocationCoordinate2DIsValid(coordinate) else { continue }
                        if let existing = combinedBounds {
                            combinedBounds = existing.includingCoordinate(coordinate)
                        } else {
                            combinedBounds = GMSCoordinateBounds(coordinate: coordinate, coordinate: coordinate)
                        }
                    }
                }

                if let combinedBounds {
                    lastFittedPairKey = pairKey
                    animateCameraFit(combinedBounds, insets: insets, on: mapView)
                    return
                }
            }

            guard let source, let destination,
                  CLLocationCoordinate2DIsValid(source.coordinate),
                  CLLocationCoordinate2DIsValid(destination.coordinate) else {
                if source == nil || destination == nil {
                    lastFittedPairKey = nil
                }
                return
            }

            let pairKey = "markers:\(source.placeID)|\(destination.placeID)"
            if !force, pairKey == lastFittedPairKey { return }
            lastFittedPairKey = pairKey

            let bounds = GMSCoordinateBounds(
                coordinate: source.coordinate,
                coordinate: destination.coordinate
            )
            animateCameraFit(bounds, insets: insets, on: mapView)
        }

        private func animateCameraFit(
            _ bounds: GMSCoordinateBounds,
            insets: UIEdgeInsets,
            on mapView: GMSMapView
        ) {
            let update = GMSCameraUpdate.fit(bounds, with: insets)
            DispatchQueue.main.async { [weak mapView] in
                guard let mapView else { return }
                mapView.animate(with: update)

                // Prevent excessive zoom on very short routes while keeping the full path visible.
                DispatchQueue.main.async { [weak mapView] in
                    guard let mapView else { return }
                    let maxZoom: Float = 17.5
                    if mapView.camera.zoom > maxZoom {
                        let clamped = GMSCameraPosition(
                            target: mapView.camera.target,
                            zoom: maxZoom,
                            bearing: mapView.camera.bearing,
                            viewingAngle: mapView.camera.viewingAngle
                        )
                        mapView.animate(to: clamped)
                    }
                }
            }
        }

        // MARK: Navigation start / end

        func handleNavigationStartIfNeeded(_ requestID: UUID?) {
            guard let requestID, requestID != handledStartID, !isStarting else { return }
            handledStartID = requestID
            beginNavigationFlow()
        }

        func handleNavigationEndIfNeeded(_ requestID: UUID?) {
            guard let requestID, requestID != handledEndID else { return }
            handledEndID = requestID
            stopNavigation()
            notifyEnded()
        }

        func handleCameraToggleIfNeeded(_ requestID: UUID?) {
            guard let requestID, requestID != handledCameraToggleRequestID else { return }
            handledCameraToggleRequestID = requestID

            let work = { [weak self] in
                guard let self, let mapView = self.mapView else {
                    print("[MAP CONTROL] Recenter skipped — map unavailable")
                    return
                }
                guard self.mode == .navigation, mapView.isNavigationEnabled else {
                    print("[MAP CONTROL] Recenter skipped — navigation inactive")
                    return
                }

                // Restore Navigation SDK following / recenter behavior.
                mapView.cameraMode = .following
                print("[MAP CONTROL] Recenter applied — cameraMode=following")
            }

            if Thread.isMainThread {
                work()
            } else {
                DispatchQueue.main.async(execute: work)
            }
        }

        func handleMyLocationIfNeeded(_ requestID: UUID?) {
            guard let requestID, requestID != handledMyLocationRequestID else { return }
            handledMyLocationRequestID = requestID

            let work = { [weak self] in
                guard let self, let mapView = self.mapView else {
                    print("[MAP CONTROL] My Location skipped — map unavailable")
                    return
                }

                let coordinate: CLLocationCoordinate2D?
                if let myLocation = mapView.myLocation?.coordinate,
                   CLLocationCoordinate2DIsValid(myLocation) {
                    coordinate = myLocation
                } else if let managerLocation = self.locationManager.location?.coordinate,
                          CLLocationCoordinate2DIsValid(managerLocation) {
                    coordinate = managerLocation
                } else {
                    coordinate = nil
                }

                guard let coordinate else {
                    print("[MAP CONTROL] My Location skipped — no valid coordinate")
                    return
                }

                let zoom = max(mapView.camera.zoom, 16.5)
                let camera = GMSCameraPosition.camera(withTarget: coordinate, zoom: zoom)
                mapView.animate(to: camera)
                print("[MAP CONTROL] My Location applied — zoom=\(zoom)")
            }

            if Thread.isMainThread {
                work()
            } else {
                DispatchQueue.main.async(execute: work)
            }
        }

        private func beginNavigationFlow() {
            #if DEBUG
            print("[NAV_GO] beginNavigationFlow mapReady=\(mapView != nil) selectedPreview=\(previewRoute?.id ?? "nil")")
            #endif
            guard let mapView else {
                notifyFailed("Map is not ready.")
                return
            }
            isStarting = true

            let options = GMSNavigationTermsAndConditionsOptions(companyName: "Wheel Buddy")
            GMSNavigationServices.showTermsAndConditionsDialogIfNeeded(with: options) { [weak self] accepted in
                Task { @MainActor in
                    guard let self else { return }
                    guard accepted else {
                        self.isStarting = false
                        self.notifyTermsRejected()
                        return
                    }
                    self.enableNavigatorAndSetDestinations(on: mapView)
                }
            }
        }

        private func enableNavigatorAndSetDestinations(on mapView: GMSMapView) {
            mapView.isNavigationEnabled = true
            mapView.travelMode = .driving
            applyDisabledGoogleNavigationControls(on: mapView)
            applyInterfaceStyle(colorScheme, navigationEnabled: true)

            guard let navigator = mapView.navigator else {
                isStarting = false
                mapView.isNavigationEnabled = false
                notifyFailed("Couldn't create a navigator.")
                return
            }

            if !didRegisterNavigatorListener {
                navigator.add(self)
                didRegisterNavigatorListener = true
            }

            let waypoints: [GMSNavigationWaypoint]
            do {
                waypoints = try Self.makeWaypoints(
                    source: source,
                    destination: destination
                )
            } catch {
                isStarting = false
                mapView.isNavigationEnabled = false
                notifyFailed(error.localizedDescription)
                return
            }

            navigator.setDestinations(waypoints) { [weak self] routeStatus in
                Task { @MainActor in
                    self?.handleRouteStatus(routeStatus, mapView: mapView, navigator: navigator)
                }
            }
        }

        private func handleRouteStatus(
            _ routeStatus: GMSRouteStatus,
            mapView: GMSMapView,
            navigator: GMSNavigator
        ) {
            defer { isStarting = false }

            guard routeStatus == .OK else {
                mapView.isNavigationEnabled = false
                notifyFailed(Self.message(for: routeStatus))
                return
            }

            // NavigationView.swift — inside handleRouteStatus
            print("[NAV 1] routeStatus =", routeStatus.rawValue)

            navigator.isGuidanceActive = true
            print("[NAV 2] guidanceActive =", navigator.isGuidanceActive)

            notifyStarted()

            navigator.isGuidanceActive = true
            navigator.sendsBackgroundNotifications = true
            // Required to receive speeding feed callbacks (nil suppresses listener updates).
            // Uses Google sample-style thresholds; not aggressive custom tuning.
            navigator.speedAlertOptions = Self.makeDefaultSpeedAlertOptions()
            mapView.cameraMode = .following
            applyDisabledGoogleNavigationControls(on: mapView)
            applyInterfaceStyle(colorScheme, navigationEnabled: true)
            applySpeedLimitDisplay(enabled: true)
            setPreviewPolylinesVisible(false)

            notifyStarted()
        }

        private static func makeDefaultSpeedAlertOptions() -> GMSNavigationSpeedAlertOptions {
            let options = GMSNavigationMutableSpeedAlertOptions()
            // Google Navigation sample thresholds (percentage above posted limit).
            options.setSpeedAlertThresholdPercentage(0.05, for: .minor)
            options.setSpeedAlertThresholdPercentage(0.10, for: .major)
            options.severityUpgradeDurationSeconds = 5
            return options
        }

        private func stopNavigation() {
            guard let mapView else { return }
            if let navigator = mapView.navigator {
                navigator.isGuidanceActive = false
                navigator.sendsBackgroundNotifications = false
                navigator.speedAlertOptions = nil
                navigator.clearDestinations()
                if didRegisterNavigatorListener {
                    _ = navigator.remove(self)
                    didRegisterNavigatorListener = false
                }
            }
            mapView.isNavigationEnabled = false
            applySpeedLimitDisplay(enabled: false)
            publishSpeedingState(.unavailable)
            publishSpeedingPercentage(nil)
            UIApplication.shared.isIdleTimerDisabled = false
            lastFittedPairKey = nil
            isStarting = false
            clearNavigationDisplayInfo()
            updatePreviewPolylines(previewRoutes, selectedRoute: previewRoute, mode: .showRoute)
            updateMarkers(
                source: source,
                destination: destination,
                mode: .showRoute,
                previewRoutes: previewRoutes
            )
        }

        private func publishNavigationDisplayInfo(_ info: NavigationDisplayInfo) {
            guard info != lastPublishedDisplayInfo else { return }
            lastPublishedDisplayInfo = info
            let callback = onNavigationInfoUpdated
            Task { @MainActor in callback(info) }
        }

        private func clearNavigationDisplayInfo() {
            lastPublishedDisplayInfo = .empty
            let callback = onNavigationInfoCleared
            Task { @MainActor in callback() }
        }

        private func makeDisplayInfo(from navInfo: GMSNavigationNavInfo) -> NavigationDisplayInfo {
            let remainingTimeText = Self.formatRemainingTime(
                seconds: navInfo.roundedTime(navInfo.timeToFinalDestinationSeconds)
            )
            let remainingDistanceText = distanceFormatter.string(
                from: navInfo.roundedDistance(navInfo.distanceToFinalDestinationMeters)
            )

            let step = navInfo.currentStep
            let instruction = step?.fullInstructionText.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let road = step?.simpleRoadName.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let image = step?.maneuverImage(with: nil)

            return NavigationDisplayInfo(
                remainingTimeText: remainingTimeText.isEmpty ? "--" : remainingTimeText,
                remainingDistanceText: remainingDistanceText.isEmpty ? "--" : remainingDistanceText,
                instructionText: instruction.isEmpty ? "Continue" : instruction,
                roadName: road,
                maneuverImage: image
            )
        }

        private static func formatRemainingTime(seconds: TimeInterval) -> String {
            let totalMinutes = max(1, Int((seconds / 60.0).rounded()))
            if totalMinutes < 60 {
                return "\(totalMinutes) min"
            }
            let hours = totalMinutes / 60
            let minutes = totalMinutes % 60
            if minutes == 0 {
                return "\(hours) hr"
            }
            return "\(hours) hr \(minutes) min"
        }

        private func notifyStarted() {
            let callback = onNavigationStarted
            Task { @MainActor in callback() }
        }

        private func notifyFailed(_ message: String) {
            let callback = onNavigationFailed
            Task { @MainActor in callback(message) }
        }

        private func notifyTermsRejected() {
            let callback = onTermsRejected
            Task { @MainActor in callback() }
        }

        private func notifyEnded() {
            let callback = onNavigationEnded
            Task { @MainActor in callback() }
        }

        private func notifyArrived() {
            let callback = onArrived
            Task { @MainActor in callback() }
        }

        private func publishSpeedingState(_ state: SpeedingState, percentage: CGFloat? = nil) {
            guard state != lastPublishedSpeedingState else { return }
            #if DEBUG
            let pct = percentage.map { String(format: "%.3f", Double($0)) } ?? "n/a"
            print(
                "[Overspeed] \(speedingLabel(lastPublishedSpeedingState)) -> \(speedingLabel(state)) percentage=\(pct)"
            )
            #endif
            lastPublishedSpeedingState = state
            let callback = onSpeedingStateChanged
            Task { @MainActor in callback(state) }
        }

        /// Publishes raw speeding fraction on every meaningful change (for OVER LIMIT MPH).
        private func publishSpeedingPercentage(_ percentage: CGFloat?) {
            if let last = lastPublishedSpeedingPercentage, let percentage,
               abs(last - percentage) < 0.0001 {
                return
            }
            if lastPublishedSpeedingPercentage == nil, percentage == nil {
                return
            }
            lastPublishedSpeedingPercentage = percentage
            let callback = onSpeedingPercentageChanged
            Task { @MainActor in callback(percentage) }
        }

        private func speedingLabel(_ state: SpeedingState) -> String {
            switch state {
            case .unavailable: return "unavailable"
            case .normal: return "normal"
            case .minor: return "minor"
            case .major: return "major"
            }
        }

        private func mapSpeedAlertSeverity(
            _ severity: GMSNavigationSpeedAlertSeverity,
            percentageAboveLimit: CGFloat
        ) -> SpeedingState {
            switch severity {
            case .unknown:
                return .unavailable
            case .notSpeeding:
                return .normal
            case .minor:
                return .minor
            case .major:
                return .major
            @unknown default:
                if percentageAboveLimit < 0 {
                    return .unavailable
                }
                return .unavailable
            }
        }

        static func makeWaypoints(
            source: SelectedPlace?,
            destination: SelectedPlace?
        ) throws -> [GMSNavigationWaypoint] {
            guard let destination else {
                throw NavigationStartupError.message("Choose a destination first.")
            }
            let destinationWaypoint = try waypoint(for: destination)

            guard let source else {
                return [destinationWaypoint]
            }

            if source.isCurrentLocation {
                return [destinationWaypoint]
            }

            let sourceWaypoint = try waypoint(for: source)
            return [sourceWaypoint, destinationWaypoint]
        }

        private static func waypoint(for place: SelectedPlace) throws -> GMSNavigationWaypoint {
            if !place.isCurrentLocation,
               !place.placeID.isEmpty,
               let byPlaceID = GMSNavigationWaypoint(
                placeID: place.placeID,
                title: place.primaryDisplayName
               ) {
                return byPlaceID
            }

            guard CLLocationCoordinate2DIsValid(place.coordinate),
                  let byCoordinate = GMSNavigationWaypoint(
                    location: place.coordinate,
                    title: place.primaryDisplayName
                  ) else {
                throw NavigationStartupError.message("Invalid waypoint for \(place.primaryDisplayName).")
            }
            return byCoordinate
        }

        private static func message(for status: GMSRouteStatus) -> String {
            switch status {
            case .OK:
                return "Route ready."
            case .noRouteFound:
                return "No navigation route found."
            case .networkError:
                return "Network error while starting navigation."
            case .quotaExceeded:
                return "Navigation quota exceeded."
            case .apiKeyNotAuthorized:
                return "API key is not authorized for Navigation SDK."
            case .canceled:
                return "Navigation route request was canceled."
            case .locationUnavailable:
                return "Current location is unavailable."
            case .waypointError:
                return "One of the waypoints is invalid."
            case .duplicateWaypointsError:
                return "Duplicate waypoints were provided."
            case .noWaypointsError:
                return "No waypoints were provided."
            case .travelModeUnsupported:
                return "Driving mode is unsupported for this route."
            case .internalError:
                return "Navigation internal error."
            @unknown default:
                return "Couldn't start navigation (status \(status.rawValue))."
            }
        }

        // MARK: Location

        private func requestLocationAccessIfNeeded() {
            switch locationManager.authorizationStatus {
            case .notDetermined:
                locationManager.requestWhenInUseAuthorization()
            case .authorizedAlways, .authorizedWhenInUse:
                beginTracking()
            case .denied, .restricted:
                mapView?.isMyLocationEnabled = false
            @unknown default:
                break
            }
        }

        private func beginTracking() {
            mapView?.isMyLocationEnabled = true
            if let location = locationManager.location {
                moveCamera(to: location, animated: false)
            }
            locationManager.startUpdatingLocation()
        }

        private func moveCamera(to location: CLLocation, animated: Bool) {
            guard mode != .navigation,
                  sourceMarker == nil,
                  destinationMarker == nil,
                  previewPolylines.isEmpty else { return }

            let camera = GMSCameraPosition.camera(withTarget: location.coordinate, zoom: 16)
            if animated {
                mapView?.animate(to: camera)
            } else {
                mapView?.camera = camera
            }
        }

        func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
            switch manager.authorizationStatus {
            case .authorizedAlways, .authorizedWhenInUse:
                beginTracking()
            case .denied, .restricted:
                mapView?.isMyLocationEnabled = false
                manager.stopUpdatingLocation()
            default:
                break
            }
        }

        func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
            guard !didApplyLiveFix, let location = locations.last else { return }
            didApplyLiveFix = true
            moveCamera(to: location, animated: true)
            manager.stopUpdatingLocation()
        }

        func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {}

        // MARK: GMSNavigatorListener

        func navigator(_ navigator: GMSNavigator, didUpdate navInfo: GMSNavigationNavInfo) {
            switch navInfo.navState {
            case .enroute:
                if let mapView {
                    applyDisabledGoogleNavigationControls(on: mapView)
                    applySpeedLimitDisplay(enabled: true)
                }
                publishNavigationDisplayInfo(makeDisplayInfo(from: navInfo))
            case .stopped:
                clearNavigationDisplayInfo()
                publishSpeedingState(.unavailable)
                publishSpeedingPercentage(nil)
            default:
                break
            }
        }

        func navigator(
            _ navigator: GMSNavigator,
            didUpdate speedAlertSeverity: GMSNavigationSpeedAlertSeverity,
            speedingPercentage percentageAboveLimit: CGFloat
        ) {
            let mapped = mapSpeedAlertSeverity(
                speedAlertSeverity,
                percentageAboveLimit: percentageAboveLimit
            )
            publishSpeedingState(mapped, percentage: percentageAboveLimit)
            publishSpeedingPercentage(percentageAboveLimit)
        }

        func navigator(_ navigator: GMSNavigator, didArriveAt waypoint: GMSNavigationWaypoint) {
            notifyArrived()
        }

        deinit {
            if let navigator = mapView?.navigator, didRegisterNavigatorListener {
                _ = navigator.remove(self)
            }
        }
    }
}

private enum NavigationStartupError: LocalizedError {
    case message(String)

    var errorDescription: String? {
        switch self {
        case .message(let text):
            return text
        }
    }
}

#Preview {
    NavigationView(
        speedingState: .constant(.unavailable),
        percentageAboveLimit: .constant(nil)
    )
        .frame(height: 480)
        .padding()
        .preferredColorScheme(.dark)
        .background(Color.black)
}
