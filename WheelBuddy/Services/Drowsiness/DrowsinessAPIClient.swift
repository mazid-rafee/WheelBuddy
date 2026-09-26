//
//  DrowsinessAPIClient.swift
//  WheelBuddy
//

import Foundation

enum DrowsinessAPIClientError: LocalizedError, Sendable {
    case invalidResponse
    case httpStatus(Int, String)
    case apiError(code: String, message: String)
    case decoding(Error)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Drowsiness API returned an invalid response."
        case let .httpStatus(code, body):
            return "Drowsiness API HTTP \(code): \(body)"
        case let .apiError(code, message):
            return "Drowsiness API error (\(code)): \(message)"
        case let .decoding(error):
            return "Drowsiness API decoding failed: \(error.localizedDescription)"
        }
    }
}

final class DrowsinessAPIClient: Sendable {
    private let session: URLSession
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    init(session: URLSession? = nil) {
        if let session {
            self.session = session
        } else {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.timeoutIntervalForRequest = 4
            configuration.timeoutIntervalForResource = 6
            configuration.waitsForConnectivity = false
            self.session = URLSession(configuration: configuration)
        }

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        self.encoder = encoder
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        self.decoder = decoder
    }

    func predict(_ request: DrowsinessPredictRequest) async throws -> DrowsinessPredictResponse {
        var urlRequest = URLRequest(url: DrowsinessAPIConfiguration.predictURL)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.timeoutInterval = 4
        if let key = DrowsinessAPIConfiguration.apiKey {
            urlRequest.setValue(key, forHTTPHeaderField: "X-API-Key")
        }
        urlRequest.httpBody = try encoder.encode(request)

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: urlRequest)
        } catch is CancellationError {
            throw CancellationError()
        } catch let urlError as URLError where urlError.code == .cancelled {
            throw CancellationError()
        }

        guard let http = response as? HTTPURLResponse else {
            throw DrowsinessAPIClientError.invalidResponse
        }

        guard (200..<300).contains(http.statusCode) else {
            if let envelope = try? decoder.decode(DrowsinessAPIErrorEnvelope.self, from: data) {
                throw DrowsinessAPIClientError.apiError(
                    code: envelope.error.code,
                    message: envelope.error.message
                )
            }
            let body = String(data: data, encoding: .utf8) ?? "<non-utf8 body>"
            throw DrowsinessAPIClientError.httpStatus(http.statusCode, body)
        }

        do {
            return try decoder.decode(DrowsinessPredictResponse.self, from: data)
        } catch {
            throw DrowsinessAPIClientError.decoding(error)
        }
    }
}
