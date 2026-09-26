//
//  DirectionsSearchCard.swift
//  WheelBuddy
//

import SwiftUI

enum DirectionsSearchField: Hashable {
    case source
    case destination
}

/// Google Maps–style source / destination card with editable TextFields.
struct DirectionsSearchCard: View {
    @Binding var sourceText: String
    @Binding var destinationText: String
    var focusedField: FocusState<DirectionsSearchField?>.Binding

    var onSourceTextChange: (String) -> Void
    var onDestinationTextChange: (String) -> Void
    var onClearSource: () -> Void
    var onClearDestination: () -> Void
    var onSwap: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 6) {
            VStack(spacing: 0) {
                fieldRow(
                    field: .source,
                    text: $sourceText,
                    placeholder: "Choose starting point",
                    indicatorColor: .blue,
                    accessibilityLabel: "Starting point",
                    onClear: onClearSource
                )

                Divider()
                    .padding(.leading, 22)

                fieldRow(
                    field: .destination,
                    text: $destinationText,
                    placeholder: "Choose destination",
                    indicatorColor: .red,
                    accessibilityLabel: "Destination",
                    onClear: onClearDestination
                )
            }

            Button(action: onSwap) {
                Image(systemName: "arrow.up.arrow.down")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 36, minHeight: 36)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Swap starting point and destination")
        }
        .padding(.leading, 10)
        .padding(.trailing, 4)
        .padding(.vertical, 4)
        .background {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.systemBackground))
                .shadow(color: Color.black.opacity(0.18), radius: 8, x: 0, y: 3)
        }
        .accessibilityElement(children: .contain)
        // Observe the bound strings directly so each keystroke is delivered immediately.
        .onChange(of: sourceText) { _, newValue in
            onSourceTextChange(newValue)
        }
        .onChange(of: destinationText) { _, newValue in
            onDestinationTextChange(newValue)
        }
    }

    private func fieldRow(
        field: DirectionsSearchField,
        text: Binding<String>,
        placeholder: String,
        indicatorColor: Color,
        accessibilityLabel: String,
        onClear: @escaping () -> Void
    ) -> some View {
        let isActive = focusedField.wrappedValue == field
        let indicatorSize: CGFloat = isActive ? 12 : 8

        return HStack(spacing: 8) {
            Circle()
                .fill(indicatorColor)
                .frame(width: indicatorSize, height: indicatorSize)
                .animation(.easeInOut(duration: 0.15), value: isActive)
                .accessibilityHidden(true)

            TextField(placeholder, text: text)
                .font(.subheadline)
                .textInputAutocapitalization(.words)
                .disableAutocorrection(true)
                .focused(focusedField, equals: field)
                .submitLabel(.search)
                .accessibilityLabel(accessibilityLabel)

            if !text.wrappedValue.isEmpty {
                Button(action: onClear) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.subheadline)
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Clear \(accessibilityLabel.lowercased())")
            }
        }
        .padding(.vertical, 7)
        .padding(.horizontal, 2)
        .contentShape(Rectangle())
        .simultaneousGesture(TapGesture().onEnded {
            focusedField.wrappedValue = field
        })
    }
}
