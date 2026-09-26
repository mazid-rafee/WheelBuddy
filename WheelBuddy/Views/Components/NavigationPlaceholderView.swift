//
//  NavigationPlaceholderView.swift
//  WheelBuddy
//

import SwiftUI

/// Upper navigation region reserved for a future MapKit map.
/// Swap this view for a Map later without redesigning DriveView.
struct NavigationPlaceholderView: View {
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(Color.white.opacity(0.06))
                .overlay(
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
                )

            VStack(spacing: 10) {
                Image(systemName: "road.lanes")
                    .font(.system(size: 36, weight: .light))
                    .foregroundStyle(.secondary)

                Text("Navigation")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.secondary)

                Text("Map coming later")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Navigation area. Map coming later.")
    }
}

#Preview {
    NavigationPlaceholderView()
        .frame(height: 220)
        .padding()
        .preferredColorScheme(.dark)
        .background(Color.black)
}
