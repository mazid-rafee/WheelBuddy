//
//  LaneAssistView.swift
//  WheelBuddy
//
//  Compact experimental top-down lane visualization.
//

import SwiftUI

struct LaneAssistView: View {
    let result: LaneTrackingResult
    var roadHeight: CGFloat = 46

    #if DEBUG
    var debugSnapshot: LaneDebugSnapshot?
    var showDebugOverlay: Bool = false
    #endif

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height

            ZStack {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [
                                Color(red: 0.12, green: 0.13, blue: 0.15),
                                Color(red: 0.08, green: 0.09, blue: 0.10)
                            ],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )

                laneLines(width: w, height: h)
                carGlyph(width: w, height: h)

                #if DEBUG
                if showDebugOverlay, let dbg = debugSnapshot {
                    debugOverlay(dbg)
                }
                #endif
            }
            .overlay(alignment: .top) {
                Text(LaneAssistStatusText.text(for: result.state))
                    .font(.system(size: 8, weight: .bold))
                    .tracking(0.2)
                    .foregroundStyle(LaneAssistStatusText.color(for: result.state))
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, 6)
                    .padding(.top, 6)
            }
        }
        .frame(height: roadHeight)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(LaneAssistStatusText.text(for: result.state))
    }

    private func laneLines(width: CGFloat, height: CGFloat) -> some View {
        let leftEmphasis = result.state == .driftingLeft
        let rightEmphasis = result.state == .driftingRight
        let dim = result.state == .unavailable

        return ZStack {
            Path { path in
                path.move(to: CGPoint(x: width * 0.40, y: height * 0.08))
                path.addLine(to: CGPoint(x: width * 0.14, y: height * 0.94))
            }
            .stroke(
                leftEmphasis ? Color.orange : Color.white.opacity(dim ? 0.18 : 0.55),
                style: StrokeStyle(lineWidth: leftEmphasis ? 2.0 : 1.1, lineCap: .round)
            )

            Path { path in
                path.move(to: CGPoint(x: width * 0.60, y: height * 0.08))
                path.addLine(to: CGPoint(x: width * 0.86, y: height * 0.94))
            }
            .stroke(
                rightEmphasis ? Color.orange : Color.white.opacity(dim ? 0.18 : 0.55),
                style: StrokeStyle(lineWidth: rightEmphasis ? 2.0 : 1.1, lineCap: .round)
            )

            Path { path in
                path.move(to: CGPoint(x: width * 0.50, y: height * 0.12))
                path.addLine(to: CGPoint(x: width * 0.50, y: height * 0.90))
            }
            .stroke(
                Color.white.opacity(dim ? 0.08 : 0.20),
                style: StrokeStyle(lineWidth: 0.8, dash: [3, 3])
            )
        }
    }

    private func carGlyph(width: CGFloat, height: CGFloat) -> some View {
        let travel = width * 0.24
        let clamped = max(-1.1, min(1.1, result.lateralOffset))
        let x = width * 0.50 + clamped * travel
        let y = height * 0.62

        return TopDownCarView(color: carColor)
            .scaleEffect(2.0)
            .position(x: x, y: y)
            .animation(.easeOut(duration: 0.15), value: result.lateralOffset)
            .opacity(result.state == .unavailable ? 0.45 : 1.0)
    }

    private var carColor: Color {
        switch result.state {
        case .unavailable:
            return .secondary
        case .tracking:
            return Color(red: 0.55, green: 0.82, blue: 1.0)
        case .driftingLeft, .driftingRight:
            return Color.orange
        }
    }

    #if DEBUG
    private func debugOverlay(_ dbg: LaneDebugSnapshot) -> some View {
        Text(String(format: "%.2f/%.2f", Double(dbg.confidence), Double(dbg.lateralOffset)))
            .font(.system(size: 7, weight: .medium, design: .monospaced))
            .foregroundStyle(.yellow.opacity(0.9))
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .padding(2)
    }
    #endif
}

enum LaneAssistStatusText {
    static func text(for state: LaneAssistState) -> String {
        switch state {
        case .unavailable: return "Lane n/a"
        case .tracking: return "Centered"
        case .driftingLeft: return "DRIFT L"
        case .driftingRight: return "DRIFT R"
        }
    }

    static func color(for state: LaneAssistState) -> Color {
        switch state {
        case .unavailable: return .secondary
        case .tracking: return .primary.opacity(0.85)
        case .driftingLeft, .driftingRight:
            return Color(red: 0.95, green: 0.55, blue: 0.2)
        }
    }
}

/// Compact top-down car: short wide body, nose up, windshield + rear glass.
private struct TopDownCarView: View {
    var color: Color

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 3.2, style: .continuous)
                .fill(color)
                .frame(width: 12, height: 15)

            Capsule()
                .fill(color)
                .frame(width: 8, height: 3.5)
                .offset(y: -6.5)

            RoundedRectangle(cornerRadius: 1.1, style: .continuous)
                .fill(Color.black.opacity(0.38))
                .frame(width: 8, height: 3.2)
                .offset(y: -2.2)

            RoundedRectangle(cornerRadius: 1.0, style: .continuous)
                .fill(Color.black.opacity(0.24))
                .frame(width: 8, height: 2.0)
                .offset(y: 4.8)
        }
        .frame(width: 12, height: 15)
    }
}
