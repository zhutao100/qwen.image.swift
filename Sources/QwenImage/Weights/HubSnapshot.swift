import Foundation
import HuggingFace

#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

public struct HubSnapshotOptions: Sendable {
  public var repoId: String
  public var revision: String
  public var repoKind: Repo.Kind
  public var patterns: [String]
  public var cacheDirectory: URL?
  public var hfToken: String?
  public var offline: Bool
  public var useBackgroundSession: Bool

  public init(
    repoId: String,
    revision: String = "main",
    repoKind: Repo.Kind = .model,
    patterns: [String] = [],
    cacheDirectory: URL? = nil,
    hfToken: String? = nil,
    offline: Bool = false,
    useBackgroundSession: Bool = false
  ) {
    self.repoId = repoId
    self.revision = revision
    self.repoKind = repoKind
    self.patterns = patterns
    self.cacheDirectory = cacheDirectory
    self.hfToken = hfToken
    self.offline = offline
    self.useBackgroundSession = useBackgroundSession
  }
}

public struct HubSnapshotProgress: Sendable {
  public let fractionCompleted: Double
  public let completedUnitCount: Int64
  public let totalUnitCount: Int64
  public let estimatedSpeedBytesPerSecond: Double?

  public init(
    fractionCompleted: Double,
    completedUnitCount: Int64,
    totalUnitCount: Int64,
    estimatedSpeedBytesPerSecond: Double? = nil
  ) {
    self.fractionCompleted = fractionCompleted
    self.completedUnitCount = completedUnitCount
    self.totalUnitCount = totalUnitCount
    self.estimatedSpeedBytesPerSecond = estimatedSpeedBytesPerSecond
  }
}

public actor HubSnapshot {
  public typealias ProgressHandler = @Sendable (HubSnapshotProgress) -> Void

  public enum HubSnapshotError: LocalizedError {
    case invalidRepoId(String)
    case fileNotFound(String)
    case noFilesMatched([String])
    case offlineCacheMiss(String)
    case snapshotNotFound(String)

    public var errorDescription: String? {
      switch self {
      case .invalidRepoId(let repoId):
        "Invalid Hugging Face repo id: \(repoId)"
      case .fileNotFound(let relativePath):
        "File not found in snapshot: \(relativePath)"
      case .noFilesMatched(let patterns):
        if patterns.isEmpty {
          "No files matched."
        } else {
          "No files matched patterns: \(patterns.joined(separator: ", "))"
        }
      case .offlineCacheMiss(let repoId):
        "Offline mode enabled and no cached snapshot found for: \(repoId)"
      case .snapshotNotFound(let repoId):
        "Snapshot not found after download: \(repoId)"
      }
    }
  }

  private let options: HubSnapshotOptions
  private let hubClient: HubClient
  private let cache: HubCache
  private let cacheDirectory: URL
  private var cachedSnapshotURL: URL?

  public init(
    options: HubSnapshotOptions,
    hubClient: HubClient? = nil
  ) throws {
    self.options = options

    let cacheDirectory = try HubSnapshot.resolveCacheDirectory(
      requested: options.cacheDirectory,
      fileManager: FileManager.default
    )
    self.cacheDirectory = cacheDirectory
    self.cache = HubCache(cacheDirectory: cacheDirectory)

    if let hubClient {
      self.hubClient = hubClient
    } else {
      let tokenProvider = options.hfToken.map { TokenProvider.fixed(token: $0) } ?? .environment
      let session: URLSession

      #if canImport(FoundationNetworking)
      session = URLSession(configuration: .default)
      #else
      if options.useBackgroundSession {
        let identifier = "qwen.image.swift.hub.\(UUID().uuidString)"
        session = URLSession(configuration: .background(withIdentifier: identifier))
      } else {
        session = URLSession(configuration: .default)
      }
      #endif

      let endpoint: URL
      if let value = ProcessInfo.processInfo.environment["HF_ENDPOINT"],
         let url = URL(string: value) {
        endpoint = url
      } else {
        endpoint = HubClient.defaultHost
      }

      self.hubClient = HubClient(
        session: session,
        host: endpoint,
        tokenProvider: tokenProvider,
        cache: cache
      )
    }
  }

  public func prepare(progressHandler: ProgressHandler? = nil) async throws -> URL {
    if let cachedSnapshotURL,
      FileManager.default.fileExists(atPath: cachedSnapshotURL.path) {
      return cachedSnapshotURL
    }

    guard let repo = Repo.ID(rawValue: options.repoId) else {
      throw HubSnapshotError.invalidRepoId(options.repoId)
    }

    if let snapshotURL = Self.resolveHuggingFaceSnapshotURL(
      cache: cache,
      repoId: repo,
      repoKind: options.repoKind,
      revision: options.revision,
      fileManager: FileManager.default
    ) {
      cachedSnapshotURL = snapshotURL
      return snapshotURL
    }

    let legacySnapshot = cacheDirectory
      .appending(path: options.repoKind.pluralized)
      .appending(path: options.repoId)
    var isLegacyDir: ObjCBool = false
    if FileManager.default.fileExists(atPath: legacySnapshot.path, isDirectory: &isLegacyDir),
       isLegacyDir.boolValue {
      cachedSnapshotURL = legacySnapshot
      return legacySnapshot
    }

    if options.offline {
      throw HubSnapshotError.offlineCacheMiss(options.repoId)
    }

    let snapshotURL = try await downloadSnapshotToCache(
      repoId: repo,
      kind: options.repoKind,
      revision: options.revision,
      matching: options.patterns,
      progressHandler: progressHandler
    )
    cachedSnapshotURL = snapshotURL
    return snapshotURL
  }

  public func fileURL(
    for relativePath: String,
    progressHandler: ProgressHandler? = nil
  ) async throws -> URL {
    let snapshot = try await prepare(progressHandler: progressHandler)
    let url = snapshot.appending(path: relativePath)
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw HubSnapshotError.fileNotFound(relativePath)
    }
    return url
  }

  public func invalidateCache() {
    cachedSnapshotURL = nil
  }

  private static func resolveHuggingFaceSnapshotURL(
    cache: HubCache,
    repoId: Repo.ID,
    repoKind: Repo.Kind,
    revision: String,
    fileManager: FileManager
  ) -> URL? {
    // huggingface_hub cache layout:
    //   <HF_HUB_CACHE>/
    //     models--<org>--<repo>/
    //       refs/<revision>          (contains commit hash)
    //       snapshots/<commitHash>/  (materialized snapshot)
    let repoDirectory = cache.repoDirectory(repo: repoId, kind: repoKind)

    var isRepoDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: repoDirectory.path, isDirectory: &isRepoDirectory),
          isRepoDirectory.boolValue
    else {
      return nil
    }

    let commitHash: String?
    if isCommitHash(revision) {
      commitHash = revision.lowercased()
    } else {
      commitHash = resolveCommitHash(
        revision: revision,
        repoDirectory: repoDirectory,
        fileManager: fileManager
      )
    }

    guard let commitHash else { return nil }

    let snapshotDirectory = cache
      .snapshotsDirectory(repo: repoId, kind: repoKind)
      .appendingPathComponent(commitHash)

    var isSnapshotDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: snapshotDirectory.path, isDirectory: &isSnapshotDirectory),
          isSnapshotDirectory.boolValue
    else {
      return nil
    }

    return snapshotDirectory
  }

  private static func resolveCommitHash(
    revision: String,
    repoDirectory: URL,
    fileManager: FileManager
  ) -> String? {
    let refsDirectory = repoDirectory.appending(path: "refs")

    var candidates = [revision]
    if revision.hasPrefix("refs/") {
      let trimmed = String(revision.dropFirst("refs/".count))
      if !trimmed.isEmpty {
        candidates.append(trimmed)
      }
    }

    for candidate in candidates {
      let refFile = refsDirectory.appending(path: candidate)
      guard fileManager.fileExists(atPath: refFile.path) else { continue }
      guard let contents = try? String(contentsOf: refFile, encoding: .utf8) else { continue }
      let hash = contents.trimmingCharacters(in: .whitespacesAndNewlines)
      if isCommitHash(hash) {
        return hash.lowercased()
      }
    }

    return nil
  }

  private static func isCommitHash(_ value: String) -> Bool {
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.count == 40 else { return false }
    let hexSet = CharacterSet(charactersIn: "0123456789abcdefABCDEF")
    return trimmed.unicodeScalars.allSatisfy { hexSet.contains($0) }
  }

  private static func resolveCacheDirectory(
    requested: URL?,
    fileManager: FileManager
  ) throws -> URL {
    if let explicit = requested {
      try fileManager.createDirectory(at: explicit, withIntermediateDirectories: true, attributes: nil)
      return explicit
    }

    // Follow HuggingFace cache convention:
    // 1. HF_HUB_CACHE env var
    // 2. HF_HOME env var + "/hub"
    // 3. ~/.cache/huggingface/hub (default)
    let env = ProcessInfo.processInfo.environment

    if let hfHubCache = env["HF_HUB_CACHE"], !hfHubCache.isEmpty {
      let directory = URL(fileURLWithPath: hfHubCache)
      try fileManager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
      return directory
    }

    if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
      let directory = URL(fileURLWithPath: hfHome).appending(path: "hub")
      try fileManager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
      return directory
    }

    // Default: ~/.cache/huggingface/hub (standard HuggingFace location)
    let home = fileManager.homeDirectoryForCurrentUser
    let directory = home.appending(path: ".cache/huggingface/hub")
    try fileManager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
    return directory
  }

  private func downloadSnapshotToCache(
    repoId: Repo.ID,
    kind: Repo.Kind,
    revision: String,
    matching globs: [String],
    progressHandler: ProgressHandler?
  ) async throws -> URL {
    let entries = try await hubClient.listFiles(
      in: repoId,
      kind: kind,
      revision: revision,
      recursive: true
    )

    let selectedEntries = entries
      .filter { $0.type == .file }
      .filter { entry in
        guard !globs.isEmpty else { return true }
        return globs.contains { fnmatch($0, entry.path, 0) == 0 }
      }

    guard !selectedEntries.isEmpty else {
      throw HubSnapshotError.noFilesMatched(globs)
    }

    let totalBytes: Int64 = selectedEntries.reduce(into: 0) { partialResult, entry in
      partialResult += Int64(entry.size ?? 0)
    }

    func emitProgress(completedBytes: Int64, speed: Double?) {
      guard let progressHandler else { return }
      let fraction = totalBytes > 0 ? min(1.0, max(0.0, Double(completedBytes) / Double(totalBytes))) : 0
      progressHandler(
        HubSnapshotProgress(
          fractionCompleted: fraction,
          completedUnitCount: completedBytes,
          totalUnitCount: totalBytes,
          estimatedSpeedBytesPerSecond: speed
        )
      )
    }

    let tempRoot = FileManager.default.temporaryDirectory
      .appendingPathComponent("qwen.image.swift")
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempRoot) }

    var completedBytes: Int64 = 0
    emitProgress(completedBytes: completedBytes, speed: nil)

    var resolvedCommitHash: String? = Self.isCommitHash(revision) ? revision.lowercased() : nil

    for entry in selectedEntries {
      let fileSize = Int64(entry.size ?? 0)
      let effectiveRevision = resolvedCommitHash ?? revision

      if cache.cachedFilePath(repo: repoId, kind: kind, revision: effectiveRevision, filename: entry.path) != nil {
        completedBytes += fileSize
        emitProgress(completedBytes: completedBytes, speed: nil)
        continue
      }

      let destination = tempRoot.appendingPathComponent(entry.path)
      try FileManager.default.createDirectory(
        at: destination.deletingLastPathComponent(),
        withIntermediateDirectories: true
      )

      let fileProgress = Progress(totalUnitCount: fileSize > 0 ? fileSize : 0)
      let baseBytes = completedBytes

      let reporter: Task<Void, Never>?
      if let progressHandler {
        reporter = Task { [totalBytes] in
          let clock = ContinuousClock()
          var lastUpdate = clock.now
          var lastBytes = baseBytes
          var smoothedSpeed: Double?
          var lastEmission = clock.now

          func sample(at now: ContinuousClock.Instant) {
            let currentBytes = baseBytes + fileProgress.completedUnitCount
            let deltaBytes = currentBytes - lastBytes
            let duration = now - lastUpdate
            let deltaSeconds = Double(duration.components.seconds)
              + Double(duration.components.attoseconds) / 1e18

            if deltaSeconds > 0, deltaBytes > 0 {
              let currentSpeed = Double(deltaBytes) / deltaSeconds
              if let existing = smoothedSpeed {
                smoothedSpeed = existing * 0.85 + currentSpeed * 0.15
              } else {
                smoothedSpeed = currentSpeed
              }
            }

            lastUpdate = now
            lastBytes = currentBytes

            let fraction = totalBytes > 0
              ? min(1.0, max(0.0, Double(currentBytes) / Double(totalBytes)))
              : 0
            progressHandler(
              HubSnapshotProgress(
                fractionCompleted: fraction,
                completedUnitCount: currentBytes,
                totalUnitCount: totalBytes,
                estimatedSpeedBytesPerSecond: smoothedSpeed
              )
            )
          }

          sample(at: clock.now)

          while !Task.isCancelled {
            let now = clock.now
            if now - lastEmission >= .milliseconds(100) {
              sample(at: now)
              lastEmission = now
            }
            do {
              try await Task.sleep(for: .milliseconds(200))
            } catch {
              break
            }
          }
        }
      } else {
        reporter = nil
      }

      _ = try await hubClient.downloadFile(
        entry,
        from: repoId,
        to: destination,
        kind: kind,
        revision: effectiveRevision,
        progress: fileProgress
      )

      reporter?.cancel()
      if let reporter {
        await reporter.value
      }

      try? FileManager.default.removeItem(at: destination)

      let actualBytes = fileProgress.totalUnitCount > 0 ? fileProgress.totalUnitCount : fileSize
      completedBytes = baseBytes + actualBytes
      emitProgress(completedBytes: completedBytes, speed: nil)

      if resolvedCommitHash == nil,
         !Self.isCommitHash(revision),
         let commit = cache.resolveRevision(repo: repoId, kind: kind, ref: revision) {
        resolvedCommitHash = commit
      }
    }

    guard let snapshotURL = Self.resolveHuggingFaceSnapshotURL(
      cache: cache,
      repoId: repoId,
      repoKind: kind,
      revision: revision,
      fileManager: FileManager.default
    ) else {
      throw HubSnapshotError.snapshotNotFound(repoId.rawValue)
    }

    return snapshotURL
  }
}
