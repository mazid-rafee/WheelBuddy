//
//  RouteSafetyInformationDetails.swift
//  WheelBuddy
//

import Charts
import SwiftUI
import UIKit

/// Aggregated safety details for the long-press route overlay.
struct RouteSafetyDetailsModel: Equatable {
    let routeID: String
    let durationText: String
    let distanceText: String
    let safetyScore: Double?
    let safetyTierLabel: String
    let safetyTier: RouteSafetyTier?
    let hasInsufficientSafetyInfo: Bool
    /// Active 3-hour bin start hour (0, 3, …, 21) used for ranking.
    let activeHourBinStart: Int?
    let personRate: Double?
    let propertyRate: Double?
    let societyRate: Double?
    let otherRate: Double?
    /// Display safety score (`1 − adjusted risk`) keyed by training time-bin start hour.
    let safetyScoreByHourBin: [Int: Double]
    /// False when the API omitted per-bin adjusted risk (legacy server).
    let hourlySafetyChartAvailable: Bool
    let cellCount: Int
    let scoredCellCount: Int
    let outOfVocabularyCount: Int

    /// Training time-bin starts (`TIME_BIN_HOURS = 3` in CrimePredictor `time_bins.py`).
    static let hourBinStarts: [Int] = [0, 3, 6, 9, 12, 15, 18, 21]

    static func build(
        route: ComputedRoute,
        prediction: RoutePrediction?
    ) -> RouteSafetyDetailsModel {
        let tierLabel: String
        let tier: RouteSafetyTier?
        if route.hasInsufficientSafetyInfo {
            tierLabel = "Not enough safety information"
            tier = nil
        } else {
            tier = route.safetyTier
            switch route.safetyTier {
            case .safest: tierLabel = "Safest"
            case .medium: tierLabel = "Medium Risk"
            case .unsafest: tierLabel = "High Risk"
            case .none: tierLabel = "Unscored"
            }
        }

        guard let prediction else {
            return RouteSafetyDetailsModel(
                routeID: route.id,
                durationText: route.durationText,
                distanceText: route.distanceMilesText,
                safetyScore: route.safetyScore,
                safetyTierLabel: tierLabel,
                safetyTier: tier,
                hasInsufficientSafetyInfo: route.hasInsufficientSafetyInfo,
                activeHourBinStart: nil,
                personRate: nil,
                propertyRate: nil,
                societyRate: nil,
                otherRate: nil,
                safetyScoreByHourBin: [:],
                hourlySafetyChartAvailable: false,
                cellCount: 0,
                scoredCellCount: 0,
                outOfVocabularyCount: 0
            )
        }

        let chartAvailable = Self.hasAdjustedHourlyScores(prediction.timeBinScores)
        var safetyScoreByHourBin: [Int: Double] = [:]
        if chartAvailable {
            for binScore in prediction.timeBinScores {
                if let adjustedRisk = binScore.adjustedSeverityWeightedSum, adjustedRisk.isFinite {
                    safetyScoreByHourBin[binScore.hourBinStart] = 1 - adjustedRisk
                }
            }
        }

        let activeBin = prediction.activeHourBinStart
        if let routeRisk = route.safetyScore, routeRisk.isFinite {
            safetyScoreByHourBin[activeBin] = 1 - routeRisk
        }

        let activeScore = prediction.timeBinScores.first {
            $0.hourBinStart == activeBin
        }

        return RouteSafetyDetailsModel(
            routeID: route.id,
            durationText: route.durationText,
            distanceText: route.distanceMilesText,
            safetyScore: route.safetyScore,
            safetyTierLabel: tierLabel,
            safetyTier: tier,
            hasInsufficientSafetyInfo: route.hasInsufficientSafetyInfo,
            activeHourBinStart: activeBin,
            personRate: activeScore?.meanPersonRate,
            propertyRate: activeScore?.meanPropertyRate,
            societyRate: activeScore?.meanSocietyRate,
            otherRate: activeScore?.meanOtherRate,
            safetyScoreByHourBin: safetyScoreByHourBin,
            hourlySafetyChartAvailable: chartAvailable,
            cellCount: prediction.cellCount,
            scoredCellCount: prediction.scoredCellCount,
            outOfVocabularyCount: prediction.outOfVocabularyCount
        )
    }

    static func hasAdjustedHourlyScores(_ bins: [TimeBinSafetyScore]) -> Bool {
        guard bins.count == hourBinStarts.count else { return false }
        let expected = Set(hourBinStarts)
        guard Set(bins.map(\.hourBinStart)) == expected else { return false }
        return bins.allSatisfy { $0.adjustedSeverityWeightedSum?.isFinite == true }
    }
}

/// Scrollable route safety details card content.
struct RouteSafetyInformationDetails: View {
    let model: RouteSafetyDetailsModel

    private var chartRows: [(label: String, score: Double, isActive: Bool)] {
        RouteSafetyDetailsModel.hourBinStarts.compactMap { bin in
            guard let score = model.safetyScoreByHourBin[bin] else { return nil }
            return (
                Self.hourBinLabel(bin),
                score,
                model.activeHourBinStart == bin
            )
        }
    }

    /// Category order for the Y axis (midnight at top).
    private var timeLabelDomain: [String] {
        RouteSafetyDetailsModel.hourBinStarts.map(Self.hourBinLabel)
    }

    /// Same green as the GO button (`RouteSafetyStyle.safest` / `#34A853`).
    private var goGreen: Color {
        Color(uiColor: RouteSafetyStyle.baseColor(for: .safest))
    }

    private var assessmentColor: Color {
        Color(uiColor: RouteSafetyStyle.baseColor(for: model.safetyTier))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                headerSection
                tripSection
                safetySummarySection
                if !model.hasInsufficientSafetyInfo {
                    categoryRatesSection
                    if model.hourlySafetyChartAvailable {
                        hourlyChartSection
                    } else {
                        hourlyChartUnavailableSection
                    }
                }
            }
            .padding(18)
        }
        .frame(maxWidth: 440, maxHeight: 364)
        .background {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.28), radius: 18, y: 8)
        }
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Route safety details for \(model.routeID)")
    }

    private var headerSection: some View {
        Text("Route safety details")
            .font(.title3.weight(.bold))
    }

    private var tripSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("Trip")
            detailRow(label: "Time", value: model.durationText)
            detailRow(label: "Distance", value: model.distanceText)
        }
    }

    private var safetySummarySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("Safety")
            if model.hasInsufficientSafetyInfo {
                Text("Insufficient data available for reliable safety assessment")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } else {
                detailRow(
                    label: "Assessment",
                    value: model.safetyTierLabel,
                    valueColor: assessmentColor
                )
                if let routeRisk = model.safetyScore, routeRisk.isFinite {
                    detailRow(
                        label: "Safety score",
                        value: String(format: "%.2f", 1 - routeRisk)
                    )
                }
            }
        }
    }

    private var categoryRatesSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let bin = model.activeHourBinStart {
                sectionTitle("Crime rates · \(Self.hourBinLabel(bin))")
            } else {
                sectionTitle("Crime rates")
                Text("No scored cells available for this route.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if model.activeHourBinStart != nil {
                VStack(spacing: 8) {
                    crimeCategoryCard(
                        title: "Offense against person",
                        examples: "Assault, homicide, robbery, sex offenses, etc.",
                        rate: model.personRate
                    )
                    crimeCategoryCard(
                        title: "Offense against property",
                        examples: "Theft, burglary, vandalism, etc.",
                        rate: model.propertyRate
                    )
                    crimeCategoryCard(
                        title: "Offense against society",
                        examples: "Drugs, shootings, etc.",
                        rate: model.societyRate
                    )
                    crimeCategoryCard(
                        title: "Other",
                        examples: "",
                        rate: model.otherRate
                    )
                }
            }
        }
    }

    private func crimeCategoryCard(
        title: String,
        examples: String,
        rate: Double?
    ) -> some View {
        Grid(horizontalSpacing: 8, verticalSpacing: 0) {
            GridRow(alignment: .center) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.primary)
                        
                    Text(examples)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .gridCellColumns(3)

                VStack(alignment: .trailing, spacing: 2) {
                    Text(rate.map { String(format: "%.4g", $0) } ?? "—")
                        .font(.caption.weight(.semibold))
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                        .multilineTextAlignment(.trailing)
                    Text("crimes/hr")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.trailing)
                }
                .frame(maxWidth: .infinity, alignment: .trailing)
                .gridCellColumns(2)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        }
        .overlay {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.06), lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(
            "\(title), \(rate.map { String(format: "%.4g crimes/hr", $0) } ?? "unavailable")"
        )
    }

    private var hourlyChartUnavailableSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("Safety score by hour")
            Text("Hourly safety chart is unavailable until CrimePredictor returns adjusted time-bin scores.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var hourlyChartSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("Safety score by hour")

            Chart(chartRows, id: \.label) { row in
                BarMark(
                    x: .value("Safety score", row.score),
                    y: .value("Time", row.label),
                    height: .ratio(0.78)
                )
                .foregroundStyle(
                    row.isActive
                        ? goGreen
                        : goGreen.opacity(row.score > 0 ? 0.45 : 0.15)
                )
                // Rounds the bar ends; for horizontal bars this rounds the left (origin) edge.
                .cornerRadius(8, style: .continuous)
            }
            // First domain value sits at the bottom; reverse so midnight is at the top.
            .chartYScale(domain: Array(timeLabelDomain.reversed()))
            .chartXAxis {
                AxisMarks(position: .bottom, values: .automatic(desiredCount: 4)) { _ in
                    AxisGridLine()
                    AxisValueLabel()
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading, values: timeLabelDomain) { value in
                    AxisValueLabel {
                        if let label = value.as(String.self) {
                            Text(label)
                                .font(.caption2)
                                .lineLimit(1)
                                .minimumScaleFactor(0.75)
                        }
                    }
                }
            }
            .frame(height: 210)
        }
    }

    private func sectionTitle(_ text: String) -> some View {
        Text(text)
            .font(.subheadline.weight(.semibold))
    }

    private func detailRow(
        label: String,
        value: String,
        valueColor: Color = .primary
    ) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Spacer(minLength: 8)
            Text(value)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(valueColor)
                .multilineTextAlignment(.trailing)
        }
    }

    static func hourBinLabel(_ binStart: Int) -> String {
        let end = (binStart + 3) % 24
        return "\(formattedHour(binStart)) - \(formattedHour(end))"
    }

    private static func formattedHour(_ hour: Int) -> String {
        let h = ((hour % 24) + 24) % 24
        let period = h < 12 ? "AM" : "PM"
        let twelve = h % 12 == 0 ? 12 : h % 12
        return "\(twelve) \(period)"
    }
}

/// UIKit backdrop so taps register above `GMSMapView` (SwiftUI gestures often miss).
private struct RouteSafetyDismissBackdrop: UIViewRepresentable {
    let onTap: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onTap: onTap)
    }

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = UIColor.black.withAlphaComponent(0.72)
        view.isUserInteractionEnabled = true
        let recognizer = UITapGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleTap)
        )
        recognizer.cancelsTouchesInView = true
        view.addGestureRecognizer(recognizer)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.onTap = onTap
    }

    final class Coordinator: NSObject {
        var onTap: () -> Void

        init(onTap: @escaping () -> Void) {
            self.onTap = onTap
        }

        @objc func handleTap() {
            onTap()
        }
    }
}

/// Dimmed full-screen host that dismisses when the outside area is tapped.
struct RouteSafetyInformationDetailsOverlay: View {
    let model: RouteSafetyDetailsModel
    let onDismiss: () -> Void

    var body: some View {
        ZStack(alignment: .center) {
            RouteSafetyDismissBackdrop(onTap: onDismiss)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .ignoresSafeArea()
                .accessibilityLabel("Dismiss route safety details")
                .accessibilityAddTraits(.isButton)

            RouteSafetyInformationDetails(model: model)
                .padding(.horizontal, 20)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        .ignoresSafeArea()
        .allowsHitTesting(true)
        .transition(.opacity)
    }
}
