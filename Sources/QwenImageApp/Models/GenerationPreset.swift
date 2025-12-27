import Foundation

enum GenerationPreset: String, CaseIterable {
    case quick = "Quick"
    case balanced = "Balanced"
    case quality = "High Quality"
    case custom = "Custom"

    var layers: Int {
        switch self {
        case .quick: return 2
        case .balanced: return 4
        case .quality: return 6
        case .custom: return 4
        }
    }

    var steps: Int {
        switch self {
        case .quick: return 4
        case .balanced: return 20
        case .quality: return 50
        case .custom: return 20
        }
    }

    var resolution: Int {
        switch self {
        case .quick: return 640
        case .balanced: return 640
        case .quality: return 1024
        case .custom: return 640
        }
    }

    var cfgScale: Float {
        switch self {
        case .quick: return 1.0
        case .balanced: return 4.0
        case .quality: return 6.0
        case .custom: return 4.0
        }
    }

    var estimatedMinutes: Int {
        let baseSeconds = Double(steps) * 3.0
        let resolutionMultiplier = resolution == 1024 ? 4.0 : 1.0
        return Int((baseSeconds * resolutionMultiplier) / 60) + 1
    }

    var description: String {
        switch self {
        case .quick:
            return "Fast results using lightning model"
        case .balanced:
            return "Good balance of speed and quality for most images"
        case .quality:
            return "Best results for complex images (slower)"
        case .custom:
            return "Use your own custom settings"
        }
    }

    var iconName: String {
        switch self {
        case .quick: return "hare.fill"
        case .balanced: return "tortoise.fill"
        case .quality: return "star.fill"
        case .custom: return "slider.horizontal.3"
        }
    }

    var timeDisplay: String {
        if estimatedMinutes < 1 {
            return "< 1 min"
        } else if estimatedMinutes == 1 {
            return "~1 min"
        } else {
            return "~\(estimatedMinutes) mins"
        }
    }
}

extension GenerationPreset {
    func matches(layers: Int, steps: Int, resolution: Int, cfgScale: Float) -> Bool {
        guard self != .custom else { return false }
        return self.layers == layers &&
               self.steps == steps &&
               self.resolution == resolution &&
               self.cfgScale == cfgScale
    }

    @MainActor
    func apply(to viewModel: LayeredViewModel) {
        viewModel.layers = self.layers
        viewModel.steps = self.steps
        viewModel.resolution = self.resolution
        viewModel.trueCFGScale = self.cfgScale
    }
}
