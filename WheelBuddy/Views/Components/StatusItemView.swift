//
//  StatusItemView.swift
//  WheelBuddy
//

import SwiftUI

struct StatusItemView: View {
    enum Tone {
        case normal
        case caution
        case urgent
        case inactive
    }

    let title: String
    let value: String
    let tone: Tone

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2.weight(.semibold))
                .tracking(0.8)
                .foregroundStyle(.secondary)

            HStack(spacing: 6) {
                Circle()
                    .fill(dotColor)
                    .frame(width: 7, height: 7)

                Text(value)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(valueColor)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.white.opacity(0.06))
        )
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
    HStack {
        StatusItemView(title: "DRIVER", value: "Attentive", tone: .normal)
        StatusItemView(title: "ROAD", value: "Monitoring", tone: .normal)
    }
    .padding()
    .preferredColorScheme(.dark)
    .background(Color.black)
}
