//
//  DrowsinessAPIConfiguration.swift
//  WheelBuddy
//

import Foundation

enum DrowsinessAPIConfiguration {
    static let baseURLInfoKey = "DROWSINESS_API_BASE_URL"
    static let apiKeyInfoKey = "DROWSINESS_API_KEY"

    static var baseURL: URL {
        if let configured = Bundle.main.object(forInfoDictionaryKey: baseURLInfoKey) as? String {
            let trimmed = configured.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty, let url = URL(string: trimmed) {
                return url
            }
        }
        #if targetEnvironment(simulator)
        return URL(string: "http://127.0.0.1:8001")!
        #else
        assertionFailure(
            "DROWSINESS_API_BASE_URL is missing. Set it in WheelBuddy.xcconfig to your Mac LAN IP, e.g. http://192.168.x.x:8001"
        )
        return URL(string: "http://127.0.0.1:8001")!
        #endif
    }

    static var predictURL: URL {
        baseURL
            .appendingPathComponent("v1")
            .appendingPathComponent("drowsiness")
            .appendingPathComponent("predict")
    }

    static var healthURL: URL {
        baseURL.appendingPathComponent("health")
    }

    /// Optional API key from Info.plist / Secrets.xcconfig.
    static var apiKey: String? {
        guard let raw = Bundle.main.object(forInfoDictionaryKey: apiKeyInfoKey) as? String else {
            return nil
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed.hasPrefix("REPLACE_WITH") {
            return nil
        }
        return trimmed
    }
}
