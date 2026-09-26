//
//  CrimeAPIConfiguration.swift
//  WheelBuddy
//

import Foundation

/// Resolves the local CrimePredictor FastAPI base URL from build settings / Info.plist.
enum CrimeAPIConfiguration {
    /// Info.plist key populated from the `CRIME_API_BASE_URL` build setting.
    static let infoPlistKey = "CRIME_API_BASE_URL"

    /// Base URL for `POST /predict-routes` (no trailing slash).
    static var baseURL: URL {
        if let configured = Bundle.main.object(forInfoDictionaryKey: infoPlistKey) as? String {
            let trimmed = configured.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty, let url = URL(string: trimmed) {
                return url
            }
        }

        #if targetEnvironment(simulator)
        return URL(string: "http://127.0.0.1:8000")!
        #else
        assertionFailure(
            "CRIME_API_BASE_URL is missing. Set it in WheelBuddy.xcconfig to your Mac LAN IP, e.g. http://192.168.x.x:8000"
        )
        return URL(string: "http://127.0.0.1:8000")!
        #endif
    }

    static var predictRoutesURL: URL {
        baseURL.appendingPathComponent("predict-routes")
    }
}
