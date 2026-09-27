import SwiftUI

/// App entry point. `DriveView` is the single root screen.
@main
struct WheelBuddyApp: App {
    /// Kept for its launch side effects (Google SDK key registration); not referenced directly.
    @UIApplicationDelegateAdaptor(AppDelegate.self)
    private var appDelegate

    var body: some Scene {
        WindowGroup {
            DriveView()
        }
    }
}
