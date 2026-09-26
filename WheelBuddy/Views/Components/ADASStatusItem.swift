//
//  ADASStatusItem.swift
//  WheelBuddy
//

import SwiftUI

/// Compact reusable ADAS status chip for the lower telemetry strip.
/// Designed so additional modules (lane, side awareness, etc.) can be added later
/// without rewriting DriveView layout.
struct ADASStatusItem: View {
    enum Tone {
        case normal
        case caution
        case urgent
        case inactive
    }

    var icon: String?
    let title: String
    let status: String
    let tone: Tone

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 4) {
                if let icon {
                    Image(systemName: icon)
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                }

                Text(title)
                    .font(.caption2.weight(.semibold))
                    .tracking(0.7)
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 5) {
                Circle()
                    .fill(dotColor)
                    .frame(width: 6, height: 6)

                Text(status)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(valueColor)
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.white.opacity(0.06))
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(title), \(status)")
    }

    private var dotColor: Color {
        switch tone {
        case .normal:
            return .green
        case .caution:
            return .orange
        case .urgent:
            return .red
        case .inactive:
            return .secondary
        }
    }

    private var valueColor: Color {
        switch tone {
        case .normal, .inactive:
            return .primary
        case .caution:
            return .orange
        case .urgent:
            return .red
        }
    }
}

#Preview {
    HStack(spacing: 6) {
        ADASStatusItem(icon: "person.fill", title: "DRIVER", status: "Attentive", tone: .normal)
        ADASStatusItem(icon: "car.fill", title: "ROAD", status: "Clear", tone: .normal)
    }
    .padding()
    .preferredColorScheme(.dark)
    .background(Color.black)
}
