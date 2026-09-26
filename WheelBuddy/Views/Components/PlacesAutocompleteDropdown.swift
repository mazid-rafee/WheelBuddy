//
//  PlacesAutocompleteDropdown.swift
//  WheelBuddy
//

import SwiftUI

/// Google Maps–style autocomplete results floating under the search card.
struct PlacesAutocompleteDropdown: View {
    var field: DirectionsSearchField
    var predictions: [PlacePredictionItem]
    var isLoading: Bool
    var errorMessage: String?
    var showCurrentLocationOption: Bool
    var currentLocationAvailable: Bool
    var locationDeniedMessage: String?
    var onSelectPrediction: (PlacePredictionItem) -> Void
    var onSelectCurrentLocation: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            if isLoading {
                HStack(spacing: 10) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Searching…")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
            }

            if let errorMessage, !errorMessage.isEmpty {
                Text(errorMessage)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
            }

            if let locationDeniedMessage, field == .source, !locationDeniedMessage.isEmpty {
                Text(locationDeniedMessage)
                    .font(.footnote)
                    .foregroundStyle(.orange)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
            }

            ScrollView {
                LazyVStack(spacing: 0) {
                    if showCurrentLocationOption {
                        currentLocationRow
                        if !predictions.isEmpty || isLoading {
                            Divider().padding(.leading, 44)
                        }
                    }

                    ForEach(Array(predictions.enumerated()), id: \.element.id) { index, item in
                        predictionRow(item)
                        if index < predictions.count - 1 {
                            Divider().padding(.leading, 44)
                        }
                    }

                    if !isLoading
                        && errorMessage == nil
                        && predictions.isEmpty
                        && !showCurrentLocationOption {
                        Text("No places found")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 14)
                    }
                }
            }
            .frame(maxHeight: 240)
        }
        .background {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(.systemBackground))
                .shadow(color: Color.black.opacity(0.18), radius: 10, x: 0, y: 4)
        }
        // Ensure map underneath does not steal taps from the dropdown.
        .contentShape(Rectangle())
        .allowsHitTesting(true)
    }

    private var currentLocationRow: some View {
        Button(action: onSelectCurrentLocation) {
            HStack(alignment: .center, spacing: 12) {
                Image(systemName: "location.fill")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.blue)
                    .frame(width: 22)

                VStack(alignment: .leading, spacing: 2) {
                    Text("Your location")
                        .font(.body.weight(.medium))
                        .foregroundStyle(.primary)
                    Text(currentLocationAvailable ? "Use current position" : "Enable location access")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 0)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Your location")
    }

    private func predictionRow(_ item: PlacePredictionItem) -> some View {
        Button {
            onSelectPrediction(item)
        } label: {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "mappin.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.secondary)
                    .frame(width: 22)

                VStack(alignment: .leading, spacing: 2) {
                    Text(item.primaryText)
                        .font(.body.weight(.medium))
                        .foregroundStyle(.primary)
                        .multilineTextAlignment(.leading)

                    if !item.secondaryText.isEmpty {
                        Text(item.secondaryText)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.leading)
                    }
                }

                Spacer(minLength: 0)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(item.primaryText), \(item.secondaryText)")
    }
}
