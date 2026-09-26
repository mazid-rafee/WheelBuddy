//
//  WarningBannerView.swift
//  WheelBuddy
//

import SwiftUI

struct WarningBannerView: View {
    enum Style {
        case urgent
        case caution
        case critical
        /// Drowsiness closed-eyes alert.
        case wakeUp
    }

    let title: String
    let style: Style

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: iconName)
                .font(.subheadline.weight(.semibold))

            Text(title)
                .font(.subheadline.weight(.bold))
                .tracking(0.4)
                .lineLimit(1)
                .minimumScaleFactor(0.85)
        }
        .foregroundStyle(foreground)
        .frame(maxWidth: .infinity)
        .padding(.vertical, 7)
        .padding(.horizontal, 12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(background)
        )
        .accessibilityAddTraits(.isHeader)
    }

    private var iconName: String {
        switch style {
        case .urgent, .critical, .wakeUp:
            return "exclamationmark.triangle.fill"
        case .caution:
            return "person.crop.circle.badge.exclamationmark"
        }
    }

    private var foreground: Color {
        switch style {
        case .urgent, .critical, .wakeUp:
            return .black
        case .caution:
            return .primary
        }
    }

    private var background: Color {
        switch style {
        case .urgent, .wakeUp:
            return Color(uiColor: color.midRisk)
        case .critical:
            return Color(uiColor: color.highRisk)
        case .caution:
            return Color.white.opacity(0.12)
        }
    }
}

#Preview {
    VStack(spacing: 12) {
        WarningBannerView(title: "WATCH THE ROAD", style: .urgent)
        WarningBannerView(title: "VEHICLE CLOSING", style: .critical)
        WarningBannerView(title: "Wake up", style: .wakeUp)
        WarningBannerView(title: "Driver not detected", style: .caution)
    }
    .padding()
    .preferredColorScheme(.dark)
    .background(Color.black)
}
