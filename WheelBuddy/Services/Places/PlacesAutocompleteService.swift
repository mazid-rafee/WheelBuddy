//
//  PlacesAutocompleteService.swift
//  WheelBuddy
//

import Combine
import CoreLocation
import Foundation
import GooglePlacesSwift

/// Debounced Google Places autocomplete + place-details fetch.
@MainActor
final class PlacesAutocompleteService: ObservableObject {
    @Published private(set) var predictions: [PlacePredictionItem] = []
    @Published private(set) var isLoading = false
    @Published private(set) var errorMessage: String?

    private let client = PlacesClient.shared
    /// Groups autocomplete requests with the following place-details fetch for Places billing.
    private var sessionToken = AutocompleteSessionToken()
    private var searchTask: Task<Void, Never>?
    /// Incremented per `scheduleSearch`; results from older generations are discarded so a slow
    /// earlier response can't overwrite newer predictions.
    private var searchGeneration = 0

    private let debounceNanoseconds: UInt64 = 300_000_000 // 300 ms
    private let maxPredictions = 5
    private let biasRadiusMeters: CLLocationDistance = 50_000

    // MARK: - Public API

    /// Cancels any pending search and clears predictions, loading, and error state.
    /// Does not start a new billing session; call `resetSession()` for that.
    func clearResults() {
        searchTask?.cancel()
        searchTask = nil
        predictions = []
        isLoading = false
        errorMessage = nil
    }

    func presentError(_ message: String) {
        errorMessage = message
    }

    func resetSession() {
        sessionToken = AutocompleteSessionToken()
    }

    /// Debounced autocomplete. Pass an empty query to clear predictions.
    /// `biasCoordinate`, when valid, biases results toward a 50 km radius around it.
    func scheduleSearch(query: String, biasCoordinate: CLLocationCoordinate2D?) {
        searchTask?.cancel()
        searchGeneration += 1
        let generation = searchGeneration

        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            predictions = []
            isLoading = false
            errorMessage = nil
            return
        }

        isLoading = true
        errorMessage = nil

        searchTask = Task { [weak self] in
            guard let self else { return }
            do {
                try await Task.sleep(nanoseconds: self.debounceNanoseconds)
            } catch {
                return
            }
            guard !Task.isCancelled, generation == self.searchGeneration else { return }
            await self.performSearch(query: trimmed, biasCoordinate: biasCoordinate, generation: generation)
        }
    }

    /// Fetches ID, name, address, and coordinate for a chosen prediction, then ends the session.
    /// - Throws: `PlacesAutocompleteError.incompletePlaceDetails` if ID, name, or a valid coordinate
    ///   is missing (address falls back to the name), or the Places SDK error on failure.
    func fetchSelectedPlace(placeID: String) async throws -> SelectedPlace {
        let request = FetchPlaceRequest(
            placeID: placeID,
            placeProperties: [
                .placeID,
                .displayName,
                .formattedAddress,
                .coordinate
            ],
            sessionToken: sessionToken
        )

        let result = await client.fetchPlace(with: request)
        // Conclude the autocomplete billing session after details.
        resetSession()

        switch result {
        case .success(let place):
            let coordinate = place.location
            guard let id = place.placeID,
                  let name = place.displayName,
                  CLLocationCoordinate2DIsValid(coordinate) else {
                throw PlacesAutocompleteError.incompletePlaceDetails
            }
            let address = place.formattedAddress ?? name
            return SelectedPlace(
                placeID: id,
                primaryDisplayName: name,
                formattedAddress: address,
                coordinate: coordinate
            )
        case .failure(let error):
            throw error
        }
    }

    // MARK: - Networking

    /// Runs one autocomplete request and publishes up to `maxPredictions` place suggestions
    /// (non-place suggestions are dropped). No-ops if a newer search has started.
    private func performSearch(
        query: String,
        biasCoordinate: CLLocationCoordinate2D?,
        generation: Int
    ) async {
        var filter: AutocompleteFilter?
        if let biasCoordinate, CLLocationCoordinate2DIsValid(biasCoordinate) {
            filter = AutocompleteFilter(
                origin: CLLocation(
                    latitude: biasCoordinate.latitude,
                    longitude: biasCoordinate.longitude
                ),
                coordinateRegionBias: CircularCoordinateRegion(
                    center: biasCoordinate,
                    radius: biasRadiusMeters
                )
            )
        }

        let request = AutocompleteRequest(
            query: query,
            sessionToken: sessionToken,
            filter: filter
        )

        let result = await client.fetchAutocompleteSuggestions(with: request)
        guard generation == searchGeneration else { return }

        isLoading = false

        switch result {
        case .success(let suggestions):
            let mapped: [PlacePredictionItem] = suggestions.compactMap { suggestion in
                guard case .place(let place) = suggestion else { return nil }
                return PlacePredictionItem(
                    placeID: place.placeID,
                    primaryText: String(place.attributedPrimaryText.characters),
                    secondaryText: place.attributedSecondaryText.map { String($0.characters) } ?? ""
                )
            }
            predictions = Array(mapped.prefix(maxPredictions))
            errorMessage = nil
        case .failure(let error):
            predictions = []
            errorMessage = error.localizedDescription
        }
    }
}

enum PlacesAutocompleteError: LocalizedError {
    case incompletePlaceDetails

    var errorDescription: String? {
        switch self {
        case .incompletePlaceDetails:
            return "Couldn't load place details. Try another result."
        }
    }
}
