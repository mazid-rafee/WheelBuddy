import UIKit
import GoogleMaps
import GooglePlaces
import GoogleNavigation

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
