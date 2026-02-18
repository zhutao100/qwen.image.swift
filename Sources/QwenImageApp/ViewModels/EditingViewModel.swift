import Foundation
import SwiftUI
import QwenImage
import MLX
import CoreGraphics

@Observable @MainActor
final class EditingViewModel {
  var referenceImages: [NSImage] = []
  var prompt: String = ""
  var negativePrompt: String = ""

  var width: Int = 1024
  var height: Int = 1024
  var useCustomSize: Bool = false
  var editResolution: Int = 1024
  var steps: Int = 4
  var guidanceScale: Float = 1.0
  var trueCFGScale: Float = 1.0
  var seed: UInt64? = nil
  var useRandomSeed: Bool = true

  var showAdvancedOptions: Bool = false
  var selectedLoRAPath: URL? = nil
  var loraScale: Float = 1.0

  var editedImage: NSImage?
  var generationState: GenerationState = .idle {
    didSet {
      appState?.setGenerationState(generationState, for: .editing)
    }
  }
  private var generationTask: Task<Void, Never>?
  var appState: AppState?

  init() {
    selectedLoRAPath = kDefaultLightningLoRAPath
  }

  var canAddMoreReferences: Bool {
    referenceImages.count < 2
  }

  func addReferenceImage(_ image: NSImage) {
    guard canAddMoreReferences else { return }
    referenceImages.append(image)
  }

  func removeReferenceImage(at index: Int) {
    guard index < referenceImages.count else { return }
    referenceImages.remove(at: index)
  }

  func clearReferenceImages() {
    referenceImages.removeAll()
  }

  func generate() {
    guard !referenceImages.isEmpty else {
      generationState = .error("Please add at least one reference image")
      return
    }

    guard !prompt.isEmpty else {
      generationState = .error("Please enter a prompt describing the edit")
      return
    }

    guard let appState else {
      generationState = .error("App state not available")
      return
    }

    guard let modelPath = appState.modelPath(for: .edit) else {
      generationState = .error("Edit model not downloaded. Please download it first.")
      return
    }

    let referenceImageData: [Data]
    do {
      referenceImageData = try referenceImages.map { try ImageIOService.pngData(from: $0) }
    } catch {
      generationState = .error(error.localizedDescription)
      return
    }
    let promptText = prompt
    let negPromptText = negativePrompt.isEmpty ? nil : negativePrompt
    let widthValue = width
    let heightValue = height
    let editRes = editResolution
    let stepCount = steps
    let guidance = guidanceScale
    let cfgScale = trueCFGScale
    let randomSeed = useRandomSeed
    let seedValue = seed
    let loraURL = selectedLoRAPath

    generationState = .loading
    editedImage = nil

    generationTask = Task.detached { [weak self] in
      do {
        let pipeline = try QwenImagePipeline.load(from: modelPath, config: .imageEditing)

        if let url = loraURL {
          pipeline.setPendingLora(from: url, scale: 1.0)
        }

        let cgImages = try referenceImageData.map { try ImageIOService.cgImage(from: $0) }

        let actualSeed = randomSeed ? UInt64.random(in: 0...UInt64.max) : seedValue
        let params = GenerationParameters(
          prompt: promptText,
          width: widthValue,
          height: heightValue,
          steps: stepCount,
          guidanceScale: guidance,
          negativePrompt: negPromptText,
          seed: actualSeed,
          trueCFGScale: cfgScale,
          editResolution: editRes
        )

        let modelConfig = QwenModelConfiguration()

        await MainActor.run { [weak self] in
          self?.generationState = .generating(step: 0, total: stepCount, progress: 0)
        }

        let pixels: MLXArray
        if cgImages.count == 1 {
          pixels = try pipeline.generateEditedPixels(
            parameters: params,
            model: modelConfig,
            referenceImage: cgImages[0],
            maxPromptLength: nil,
            seed: actualSeed
          )
        } else {
          pixels = try pipeline.generateEditedPixels(
            parameters: params,
            model: modelConfig,
            referenceImages: cgImages,
            maxPromptLength: nil,
            seed: actualSeed
          )
        }

        if Task.isCancelled {
          await MainActor.run { [weak self] in
            self?.generationState = .idle
          }
          return
        }

        let image = try pipeline.makeImage(from: pixels)
        let pngData = try ImageIOService.pngData(from: image)

        await MainActor.run { [weak self] in
          guard let self else { return }
          self.editedImage = NSImage(data: pngData)
          self.generationState = .complete
          if randomSeed {
            self.seed = actualSeed
          }
        }

      } catch {
        await MainActor.run { [weak self] in
          guard let self else { return }
          if Task.isCancelled {
            self.generationState = .idle
          } else {
            self.generationState = .error(error.localizedDescription)
          }
        }
      }
    }
  }

  func cancelGeneration() {
    generationTask?.cancel()
    generationTask = nil
    generationState = .idle
  }

  func exportImage(to url: URL) throws {
    guard let image = editedImage else {
      throw EditingError.noImageToExport
    }
    try ImageIOService.saveImage(image, to: url, format: .png)
  }

  func clear() {
    referenceImages.removeAll()
    editedImage = nil
    generationState = .idle
  }
}

enum EditingError: LocalizedError {
  case invalidReferenceImages
  case noImageToExport

  var errorDescription: String? {
    switch self {
    case .invalidReferenceImages:
      return "Failed to process reference images"
    case .noImageToExport:
      return "No image to export. Generate an edit first."
    }
  }
}
