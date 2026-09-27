//
//  DriveView.swift
//  WheelBuddy
//

import Combine
import SwiftUI

struct DriveView: View {
    @StateObject private var driverMonitor = DriverMonitor()
    @StateObject private var roadDetector = RoadDetectionService()
    @StateObject private var roadRiskAnalyzer = RoadRiskAnalyzer()
    @StateObject private var pedestrianRiskAnalyzer = PedestrianRiskAnalyzer()
    @StateObject private var speedMonitor = SpeedMonitor()
    @StateObject private var laneDetector = LaneDetectionService()
    @StateObject private var alertManager = ADASAlertManager()
    @StateObject private var drowsinessRemote = DrowsinessInferenceCoordinator()

    /// Stable owner for the non-Observable MultiCamManager + published UI status.
    @StateObject private var multiCamOwner = MultiCamSessionOwner()

    /// Drive mode is opt-in each time this view is presented.
    @State private var isDriveModeOn = false

    /// Keep driving controls hidden until the asynchronous camera start succeeds.
    private var isDriveModeReady: Bool {
        isDriveModeOn && multiCamOwner.isActive
    }

    private var driveModeButtonColor: Color {
        isDriveModeOn ? Color(red: 1, green: 0.75, blue: 0) : Color(uiColor: color.accent)
    }

    /// Bridged from Google Navigation overspeed callbacks (unavailable outside guidance).
    @State private var speedingState: SpeedingState = .unavailable
    /// Raw Navigation SDK speeding fraction; `nil` outside guidance / after reset.
    @State private var percentageAboveLimit: CGFloat? = nil
    /// Last limit inferred from Nav SDK + GPS; shown when live estimate is unavailable.
    @State private var lastReliableSpeedLimitMPH: Int?

    #if DEBUG
    @State private var showLaneDebugOverlay = false
    @State private var lastOverLimitLogKey: String = ""
    @State private var lastRoadDisplayLog: String = ""
    #endif

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 8) {
                topBar

                // Keep navigation visible even when driving detection is off.
                NavigationView(
                    speedingState: $speedingState,
                    percentageAboveLimit: $percentageAboveLimit
                )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                if !isDriveModeReady {
                    Button(action: startDriveMode) {
                        HStack(spacing: 10) {
                            if isDriveModeOn {
                                ProgressView()
                                    .tint(.black)
                            } else {
                                Image(systemName: "car.fill")
                                    .font(.system(size: 17, weight: .semibold))
                            }
                            Text(isDriveModeOn ? "Starting Drive Mode…" : "Start Drive")
                                .font(.system(size: 16, weight: .semibold))
                        }
                            .frame(maxWidth: .infinity)
                            .frame(height: 52)
                            .foregroundStyle(isDriveModeOn ? .black : .white)
                            .background(
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .fill(driveModeButtonColor)
                            )
                            .overlay {
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .strokeBorder(.white.opacity(0.14), lineWidth: 1)
                            }
                    }
                    .buttonStyle(.plain)
                    .allowsHitTesting(!isDriveModeOn)
                    .shadow(color: driveModeButtonColor.opacity(0.25), radius: 10, y: 4)
                    .accessibilityHint(isDriveModeOn ? "Driving services are starting" : "Starts driving detection and alerts")
                }

                if isDriveModeReady {
                    lowerTelemetry
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 6)
            .padding(.bottom, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .ignoresSafeArea(.keyboard, edges: .bottom)
        .preferredColorScheme(.dark)
        .animation(.easeInOut(duration: 0.2), value: driverMonitor.attentionState)
        .animation(.easeInOut(duration: 0.2), value: unifiedRoadDisplay)
        .animation(.easeInOut(duration: 0.2), value: laneDetector.result.state)
        .animation(.easeInOut(duration: 0.2), value: percentageAboveLimit)
        .animation(.easeInOut(duration: 0.2), value: speedMonitor.speedMPH)
        .animation(.easeInOut(duration: 0.2), value: warningBanner?.title)
        .animation(.easeInOut(duration: 0.2), value: drowsinessRemote.isWakeUpAlertActive)
        .onDisappear {
            stopDriveMode()
        }
        .onReceive(roadDetector.$detections) { detections in
            guard isDriveModeOn else { return }
            let timestamp = ProcessInfo.processInfo.systemUptime
            roadRiskAnalyzer.update(detections: detections, timestamp: timestamp)
            pedestrianRiskAnalyzer.update(detections: detections, timestamp: timestamp)
            logUnifiedRoadDisplayIfNeeded()
            syncADASAlerts()
        }
        .onChange(of: roadDetector.isModelReady) { _, ready in
            if !ready {
                roadRiskAnalyzer.reset()
                pedestrianRiskAnalyzer.reset()
            }
            syncADASAlerts()
        }
        .onChange(of: driverMonitor.attentionState) { _, _ in
            syncADASAlerts()
        }
        .onChange(of: roadRiskAnalyzer.state) { _, _ in
            logUnifiedRoadDisplayIfNeeded()
            syncADASAlerts()
        }
        .onChange(of: pedestrianRiskAnalyzer.state) { _, _ in
            logUnifiedRoadDisplayIfNeeded()
            syncADASAlerts()
        }
        .onChange(of: laneDetector.result.state) { _, _ in
            syncADASAlerts()
        }
        .onChange(of: drowsinessRemote.isWakeUpAlertActive) { wasActive, isActive in
            syncADASAlerts()
            // Beep only when the banner becomes visible (rising edge).
            if isDriveModeReady, isActive, !wasActive {
                alertManager.playWakeUpBeep()
            }
        }
        .onChange(of: speedMonitor.speedMPH) { _, _ in
            syncLaneSpeedGate()
            syncLastReliableSpeedLimit()
            logOverLimitIfNeeded()
        }
        .onChange(of: speedMonitor.hasReliableSpeed) { _, _ in
            syncLaneSpeedGate()
            syncLastReliableSpeedLimit()
        }
        .onChange(of: percentageAboveLimit) { _, _ in
            syncLastReliableSpeedLimit()
            logOverLimitIfNeeded()
        }
    }

    /// Feeds centralized alert manager from explicit state changes only.
    private func syncADASAlerts() {
        guard isDriveModeReady else { return }
        alertManager.update(
            driverAttention: driverMonitor.attentionState,
            roadRisk: roadRiskAnalyzer.state,
            pedestrianRisk: pedestrianRiskAnalyzer.state,
            laneAssist: laneDetector.result.state,
            wakeUpAlert: drowsinessRemote.isWakeUpAlertActive
        )
    }

    private func syncLaneSpeedGate() {
        guard isDriveModeOn else { return }
        laneDetector.updateSpeed(
            mph: speedMonitor.speedMPH,
            reliable: speedMonitor.hasReliableSpeed
        )
    }

    /// Reserve banner space so warning changes never resize the navigation map.
    private var lowerTelemetry: some View {
        VStack(spacing: 6) {
            ZStack {
                if let banner = warningBanner {
                    WarningBannerView(title: banner.title, style: banner.style)
                        .transition(.opacity)
                } else {
                    Label("Monitoring road", systemImage: "eye.fill")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity)
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: 36)

            instrumentClusterRow
        }
    }

    // MARK: - Top bar

    private var topBar: some View {
        HStack(alignment: .center) {
            Text("WheelBuddy")
                .font(.headline.weight(.semibold))
                .foregroundStyle(.primary)

            Spacer()

            if isDriveModeReady {
                Button(action: stopDriveMode) {
                    ZStack(alignment: .topTrailing) {
                        Image(systemName: "car.fill")
                            .font(.system(size: 12, weight: .semibold))
                    }
                    .frame(width: 64, height: 24)
                    .foregroundStyle(.white)
                    .background(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(Color(uiColor: color.red))
                    )
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Drive Mode Off")
                .accessibilityHint("Stops driving detection and alerts")
            }
        }
    }

    // MARK: - Instrument area: lane left, speed and status right

    private let laneRoadHeight: CGFloat = 184 // Four times the previous 46-point road height.

    private var instrumentClusterRow: some View {
        GeometryReader { geometry in
            let columnWidth = (geometry.size.width - 8) / 2
            HStack(alignment: .top, spacing: 8) {
                laneAssistColumn
                    .frame(width: columnWidth, height: laneRoadHeight)

                VStack(spacing: 6) {
                    speedColumn
                        .layoutPriority(1)
                    driverStatusCard
                        .fixedSize(horizontal: false, vertical: true)
                    roadStatusCard
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(width: columnWidth, height: laneRoadHeight, alignment: .top)
            }
        }
        .frame(height: laneRoadHeight)
    }

    private var speedColumn: some View {
        VStack(spacing: 4) {
            Text("SPEED / LIMIT")
                .font(.system(size: 10, weight: .bold))
                .tracking(0.4)
                .foregroundStyle(.secondary)

            (Text(speedDisplayText).foregroundColor(speedForeground)
                + Text(" / ").foregroundColor(.secondary)
                + Text(speedLimitDisplayText).foregroundColor(.primary))
                .font(.system(size: 23, weight: .semibold, design: .rounded))
                .monospacedDigit()
                .minimumScaleFactor(0.7)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.vertical, 8)
        .padding(.horizontal, 6)
        .background(Color.white.opacity(0.09), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(speedAccessibilityLabel)
    }

    @ViewBuilder
    private var laneAssistColumn: some View {
        #if DEBUG
        LaneAssistView(
            result: laneDetector.result,
            roadHeight: laneRoadHeight,
            debugSnapshot: laneDetector.debugSnapshot,
            showDebugOverlay: showLaneDebugOverlay
        )
        .onLongPressGesture(minimumDuration: 0.6) {
            showLaneDebugOverlay.toggle()
        }
        #else
        LaneAssistView(result: laneDetector.result, roadHeight: laneRoadHeight)
        #endif
    }

    private var speedDisplayText: String {
        guard speedMonitor.hasReliableSpeed, let mph = speedMonitor.speedMPH else {
            return "--"
        }
        return "\(Int(mph.rounded()))"
    }

    private var speedLimitDisplayText: String {
        guard let mph = displayedSpeedLimitMPH else { return "--" }
        return "\(mph)"
    }

    /// Live Nav estimate when speeding; otherwise the last cached limit from this session.
    private var displayedSpeedLimitMPH: Int? {
        if let fresh = estimatedSpeedLimitMPHFromNavigation {
            return fresh
        }
        return lastReliableSpeedLimitMPH
    }

    /// Inferred from GPS speed and Navigation SDK overspeed fraction (only while above limit).
    private var estimatedSpeedLimitMPHFromNavigation: Int? {
        guard speedMonitor.hasReliableSpeed,
              let mph = speedMonitor.speedMPH,
              let percentage = percentageAboveLimit,
              percentage > 0 else { return nil }
        let estimatedLimit = mph / (1.0 + Double(percentage))
        let roundedLimit = Int((estimatedLimit / 5.0).rounded() * 5.0)
        return roundedLimit > 0 ? roundedLimit : nil
    }

    private func syncLastReliableSpeedLimit() {
        guard let fresh = estimatedSpeedLimitMPHFromNavigation else { return }
        lastReliableSpeedLimitMPH = fresh
    }

    /// Whole MPH over the posted limit from Nav SDK percentage + GPS speed.
    /// `p > 0` → `S * p / (1 + p)`; `p == 0` → `0`; `p < 0` / missing → `--`.
    private var overLimitDisplayText: String {
        guard let p = percentageAboveLimit else { return "--" }
        if p < 0 { return "--" }
        if p == 0 { return "0" }
        guard speedMonitor.hasReliableSpeed, let speed = speedMonitor.speedMPH else {
            return "--"
        }
        let over = speed * Double(p) / (1.0 + Double(p))
        return "\(Int(over.rounded()))"
    }

    /// Current speed only: deeper red as MPH over the limit increases.
    private var speedForeground: Color {
        guard speedMonitor.hasReliableSpeed,
              let mph = speedMonitor.speedMPH,
              let p = percentageAboveLimit,
              p > 0 else {
            return .primary
        }
        let over = mph * Double(p) / (1.0 + Double(p))
        guard over > 0 else { return .primary }

        // 0 mph over → soft red; ≥20 mph over → deep crimson.
        let t = min(1.0, over / 20.0)
        return Color(
            red: 0.92 - 0.22 * t,
            green: 0.28 - 0.24 * t,
            blue: 0.22 - 0.16 * t
        )
    }

    private var speedAccessibilityLabel: String {
        let speedPart: String
        if speedMonitor.hasReliableSpeed, let mph = speedMonitor.speedMPH {
            speedPart = "\(Int(mph.rounded())) miles per hour"
        } else {
            speedPart = "Speed unavailable"
        }
        let limitPart: String
        if let limit = displayedSpeedLimitMPH {
            limitPart = "speed limit \(limit) miles per hour"
        } else {
            limitPart = "speed limit unavailable"
        }
        return "\(speedPart), \(limitPart)"
    }

    private func logOverLimitIfNeeded() {
        #if DEBUG
        guard let p = percentageAboveLimit else {
            let key = "nil"
            guard key != lastOverLimitLogKey else { return }
            lastOverLimitLogKey = key
            print("[OverLimit] percentage unavailable")
            return
        }
        if p < 0 {
            let key = "neg"
            guard key != lastOverLimitLogKey else { return }
            lastOverLimitLogKey = key
            print("[OverLimit] percentage=-1 unavailable")
            return
        }
        let speed = speedMonitor.hasReliableSpeed ? speedMonitor.speedMPH : nil
        let overText: String
        if p == 0 {
            overText = "0"
        } else if let speed {
            overText = "\(Int((speed * Double(p) / (1.0 + Double(p))).rounded()))"
        } else {
            overText = "n/a"
        }
        let speedText = speed.map { String(format: "%.1f", $0) } ?? "n/a"
        let key = "\(speedText)|\(String(format: "%.4f", Double(p)))|\(overText)"
        guard key != lastOverLimitLogKey else { return }
        lastOverLimitLogKey = key
        if p == 0 {
            print("[OverLimit] speedMPH=\(speedText) percentage=0.0 overMPH=0")
        } else {
            print(
                String(
                    format: "[OverLimit] speedMPH=%@ percentage=%.4f overMPH=%@",
                    speedText,
                    Double(p),
                    overText
                )
            )
        }
        #endif
    }

    // MARK: - ADAS status column

    private var driverStatusCard: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Image(systemName: "person.fill")
                    .font(.system(size: 14, weight: .bold))
                    .frame(width: 16, height: 16)
                    .foregroundStyle(.secondary)
                Text("Driver Status")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(.secondary)
            }
            Text(driverDisplayText)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(driverStatusColor)
                .lineLimit(2)
                .minimumScaleFactor(0.85)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 8)
        .padding(.vertical, 8)
        .background(Color.white.opacity(0.09), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Driver, \(driverDisplayText)")
    }

    private var driverStatusColor: Color {
        switch driverMonitor.attentionState {
        case .attentive: return Color(uiColor: color.accent)
        case .lookingAway: return Color(uiColor: color.red)
        case .noFace: return .orange
        }
    }

    /// The ROAD card presents the prioritized vehicle or pedestrian result.
    private var roadStatusCard: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Image(systemName: "road.lanes.curved.right")
                    .font(.system(size: 14, weight: .bold))
                    .frame(width: 16, height: 16)
                    .foregroundStyle(.secondary)
                Text("Road Status")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(.secondary)
            }
            Text(roadDisplayText)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(roadStatusColor)
                .lineLimit(2)
                .minimumScaleFactor(0.85)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 8)
        .padding(.vertical, 8)
        .background(Color.white.opacity(0.09), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Road, \(roadDisplayText)")
    }

    private var roadStatusColor: Color {
        switch unifiedRoadDisplay {
        case .clear, .vehicleAhead, .pedestrianAhead:
            return Color(uiColor: color.accent)
        case .unavailable, .closingVehicle:
            return .orange
        case .rapidClosing, .pedestrianClose:
            return Color(uiColor: color.red)
        }
    }

    // MARK: - Display mapping (UI only)

    /// Single ROAD presentation — both analyzers stay active; only display is prioritized.
    private var unifiedRoadDisplay: UnifiedRoadDisplayState {
        UnifiedRoadDisplayState.resolve(
            modelReady: roadDetector.isModelReady,
            modelUnavailable: roadDetector.state == .modelUnavailable,
            vehicle: roadRiskAnalyzer.state,
            pedestrian: pedestrianRiskAnalyzer.state
        )
    }

    private var driverDisplayText: String {
        switch driverMonitor.attentionState {
        case .attentive:
            return "Attentive"
        case .lookingAway:
            return "Watch road"
        case .noFace:
            return "Driver not detected"
        }
    }

    private var driverTone: ADASStatusItem.Tone {
        switch driverMonitor.attentionState {
        case .attentive:
            return .normal
        case .lookingAway:
            return .urgent
        case .noFace:
            return .caution
        }
    }

    private var roadDisplayText: String {
        unifiedRoadDisplay.statusText
    }

    private var roadTone: ADASStatusItem.Tone {
        switch unifiedRoadDisplay {
        case .unavailable:
            return .caution
        case .clear, .vehicleAhead, .pedestrianAhead:
            return .normal
        case .closingVehicle:
            return .caution
        case .rapidClosing, .pedestrianClose:
            return .urgent
        }
    }

    /// Priority: Pedestrian Close! > Rapid closing > lane drift > looking away >
    /// major speeding > wake up / no face. Closing vehicle has no dedicated banner.
    private var warningBanner: (title: String, style: WarningBannerView.Style)? {
        if roadDetector.isModelReady {
            if pedestrianRiskAnalyzer.state == .close {
                return ("PEDESTRIAN CLOSE!", .critical)
            }
            if roadRiskAnalyzer.state == .high {
                return ("VEHICLE CLOSING", .critical)
            }
        }

        switch laneDetector.result.state {
        case .driftingLeft:
            return ("LANE DRIFT LEFT", .urgent)
        case .driftingRight:
            return ("LANE DRIFT RIGHT", .urgent)
        case .unavailable, .tracking:
            break
        }

        if driverMonitor.attentionState == .lookingAway {
            return ("WATCH THE ROAD", .urgent)
        }

        if speedingState == .major {
            return ("SLOW DOWN", .urgent)
        }

        // Held 3s by the coordinator; suppressed while any higher-priority case above is active.
        if drowsinessRemote.isWakeUpAlertActive {
            return ("WAKE UP", .wakeUp)
        }

        if driverMonitor.attentionState == .noFace {
            return ("Driver not detected", .caution)
        }

        return nil
    }

    private func logUnifiedRoadDisplayIfNeeded() {
        #if DEBUG
        let text = unifiedRoadDisplay.statusText
        guard text != lastRoadDisplayLog else { return }
        if !lastRoadDisplayLog.isEmpty {
            print("[RoadDisplay] \(lastRoadDisplayLog) -> \(text)")
        }
        lastRoadDisplayLog = text
        #endif
    }

    // MARK: - Drive mode and MultiCam lifecycle

    private func startDriveMode() {
        guard !isDriveModeOn else { return }
        isDriveModeOn = true
        speedMonitor.start()
        drowsinessRemote.start()
        syncLaneSpeedGate()
        startMultiCamIfNeeded()
    }

    private func stopDriveMode() {
        guard isDriveModeOn else { return }
        isDriveModeOn = false
        alertManager.stop()
        drowsinessRemote.stop()
        stopMultiCam()
        speedMonitor.stop()
        speedingState = .unavailable
        percentageAboveLimit = nil
        lastReliableSpeedLimitMPH = nil
    }

    private func startMultiCamIfNeeded() {
        guard !multiCamOwner.hasStarted, !multiCamOwner.isStopping else { return }
        multiCamOwner.hasStarted = true
        multiCamOwner.errorMessage = nil
        let startID = UUID()
        multiCamOwner.startID = startID

        // External processing only — do NOT call legacy CameraManager start APIs.
        driverMonitor.drowsinessCoordinator = drowsinessRemote
        driverMonitor.beginExternalFrameProcessing()
        roadDetector.beginExternalFrameProcessing()
        laneDetector.beginExternalFrameProcessing()

        let manager = multiCamOwner.manager
        manager.onFrontFrame = { [weak driverMonitor] pixelBuffer in
            driverMonitor?.processExternalFrame(pixelBuffer)
        }
        manager.onRearFrame = { [weak roadDetector, weak laneDetector] pixelBuffer in
            roadDetector?.processExternalFrame(pixelBuffer)
            laneDetector?.processExternalFrame(pixelBuffer)
        }

        manager.requestAccessAndStart { result in
            guard multiCamOwner.startID == startID, isDriveModeOn else {
                if case .success = result, !isDriveModeOn {
                    manager.stop { }
                }
                return
            }
            switch result {
            case .success:
                multiCamOwner.isActive = true
                multiCamOwner.errorMessage = nil
                alertManager.start()
                syncADASAlerts()
            case .failure(let error):
                multiCamOwner.isActive = false
                multiCamOwner.errorMessage = error.localizedDescription
                multiCamOwner.startID = nil
                manager.onFrontFrame = nil
                manager.onRearFrame = nil
                driverMonitor.endExternalFrameProcessing()
                roadDetector.endExternalFrameProcessing()
                laneDetector.endExternalFrameProcessing()
                roadRiskAnalyzer.reset()
                pedestrianRiskAnalyzer.reset()
                // Restore the start button and stop services after a failed start.
                multiCamOwner.hasStarted = false
                stopDriveMode()
            }
        }
    }

    private func stopMultiCam() {
        guard !multiCamOwner.isStopping,
              multiCamOwner.hasStarted || multiCamOwner.isActive else { return }

        multiCamOwner.startID = nil
        multiCamOwner.isStopping = true

        multiCamOwner.manager.onFrontFrame = nil
        multiCamOwner.manager.onRearFrame = nil
        driverMonitor.endExternalFrameProcessing()
        roadDetector.endExternalFrameProcessing()
        laneDetector.endExternalFrameProcessing()
        roadRiskAnalyzer.reset()
        pedestrianRiskAnalyzer.reset()

        multiCamOwner.manager.stop {
            multiCamOwner.isActive = false
            multiCamOwner.hasStarted = false
            multiCamOwner.isStopping = false
            if isDriveModeOn {
                startMultiCamIfNeeded()
            }
        }
    }
}

/// Holds the shared MultiCamManager and publishes DriveView status.
@MainActor
final class MultiCamSessionOwner: ObservableObject {
    let manager = MultiCamManager()

    @Published var isActive = false
    @Published var errorMessage: String?
    var hasStarted = false
    var isStopping = false
    var startID: UUID?
}

#Preview {
    DriveView()
}
