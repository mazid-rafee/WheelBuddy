//
//  RoadDebugView.swift
//  WheelBuddy
//

import SwiftUI

/// Temporary Milestone 2 test UI for rear-camera road object detection.
struct RoadDebugView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var detector = RoadDetectionService()
    @State private var isStopping = false

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 20) {
                Text("ROAD DETECTION")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .frame(maxWidth: .infinity, alignment: .center)

                Text(modelStatusText)
                    .font(.headline)
                    .foregroundStyle(detector.isModelReady ? .green : .red)

                if !detector.isModelReady {
                    Text(detector.modelLoadMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Text("Objects detected: \(detector.detections.count)")
                    .font(.title3.weight(.semibold))

                if let error = detector.cameraError {
                    Text(error.localizedDescription)
                        .font(.subheadline)
                        .foregroundStyle(.red)
                } else if detector.isModelReady && !detector.isRunning && !isStopping {
                    Text("Starting rear camera…")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                } else if isStopping {
                    Text("Stopping rear camera…")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Divider()

                if detector.detections.isEmpty {
                    Text("NO RELEVANT OBJECTS")
                        .font(.title3.weight(.medium))
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.top, 24)
                } else {
                    ForEach(detector.detections.prefix(5)) { detection in
                        HStack {
                            Text(detection.label.uppercased())
                                .font(.body.weight(.semibold))
                            Spacer()
                            Text("\(Int((detection.confidence * 100).rounded()))%")
                                .font(.body.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Spacer()
            }
            .padding()
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") {
                        stopThenDismiss()
                    }
                    .disabled(isStopping)
                }
            }
        }
        .interactiveDismissDisabled(true)
        .onAppear {
            detector.start()
        }
        .onDisappear {
            // Defensive cleanup if the cover is torn down without Done.
            // Harmless when Done already stopped the session.
            detector.stop()
        }
    }

    private var modelStatusText: String {
        detector.isModelReady ? "Model: READY" : "Model: UNAVAILABLE"
    }

    private func stopThenDismiss() {
        guard !isStopping else { return }
        isStopping = true
        detector.stop {
            dismiss()
        }
    }
}

#Preview {
    RoadDebugView()
}
