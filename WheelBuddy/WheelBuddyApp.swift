import SwiftUI

@main
struct WheelBuddyApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self)
    private var appDelegate

    var body: some Scene {
        WindowGroup {
            DriveView()
        }
    }
}
