//
//  GoogleDurationParser.swift
//  WheelBuddy
//

import Foundation

/// Shared parser for Google Routes / protobuf-style duration strings such as `"65s"` or `"3.5s"`.
enum GoogleDurationParser {
    /// Parses a duration string ending in `s` into seconds.
    /// Returns `nil` for empty, negative, malformed, NaN, or infinite values.
    static func parseSeconds(_ raw: String?) -> Double? {
        guard let raw else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.hasSuffix("s"), trimmed.count > 1 else {
            return nil
        }

        let numberPart = String(trimmed.dropLast())
        guard !numberPart.isEmpty,
              let value = Double(numberPart),
              value.isFinite,
              value >= 0 else {
            return nil
        }
        return value
    }
}

enum RouteDepartureTimeFormatting {
    /// Formats a request departure instant as ISO-8601 UTC ending in `Z`.
    static func iso8601UTCString(from date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        return formatter.string(from: date)
    }
}
