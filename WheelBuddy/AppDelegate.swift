import UIKit
import GoogleMaps
import GooglePlaces
import GoogleNavigation

/// UIKit delegate bridged into SwiftUI via `@UIApplicationDelegateAdaptor`; exists to register the
/// Google Maps / Places API key before any map or places UI is created.
final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions:
            [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        let apiKey = Self.mapsPlacesAPIKey
        GMSServices.provideAPIKey(apiKey)
        GMSPlacesClient.provideAPIKey(apiKey)
        return true
    }

    /// Shared restricted iOS key for Maps / Places / Navigation from Secrets.xcconfig.
    /// Read from Info.plist `GMSApiKey`; an unexpanded `$(...)` build variable or the placeholder
    /// value counts as missing. Asserts in debug and returns "" in release when missing.
    private static var mapsPlacesAPIKey: String {
        let raw = Bundle.main.object(forInfoDictionaryKey: "GMSApiKey") as? String
        let key = raw?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\"", with: "")
        if let key, !key.isEmpty, !key.hasPrefix("$("), key != "REPLACE_WITH_MAPS_API_KEY" {
            return key
        }
        assertionFailure("GMSApiKey missing. Copy Secrets.example.xcconfig to Secrets.xcconfig.")
        return ""
    }
}
