import SwiftUI

// MARK: - Preset Picker

/// A user-friendly preset picker for generation settings
struct PresetPickerView: View {
    @Binding var selectedPreset: GenerationPreset
    let timeEstimate: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Preset")
                    .font(.subheadline.bold())
                Spacer()
                Text(timeEstimate)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        Capsule()
                            .fill(Color.accentColor.opacity(0.1))
                    )
            }

            HStack(spacing: 8) {
                ForEach(GenerationPreset.allCases, id: \.self) { preset in
                    PresetButton(
                        preset: preset,
                        isSelected: selectedPreset == preset
                    ) {
                        selectedPreset = preset
                    }
                }
            }
        }
    }
}

struct PresetButton: View {
    let preset: GenerationPreset
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 2) {
                Image(systemName: preset.iconName)
                    .font(.system(size: 14))
                Text(preset.rawValue)
                    .font(.caption.bold())
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 44)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(isSelected ? Color.accentColor : Color.secondary.opacity(0.1))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(.plain)
        .foregroundStyle(isSelected ? .white : .primary)
    }
}

// MARK: - Parameter Slider with Slider Labels

struct ParameterSlider: View {
  let title: String
  @Binding var value: Float
  var range: ClosedRange<Float> = 1...10
  var step: Float = 0.5
  var format: String = "%.1f"
  var description: String? = nil
  var showLabels: Bool = false
  var leftLabel: String? = nil
  var rightLabel: String? = nil

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      HStack {
        Text(title)
          .font(.subheadline.bold())
        Spacer()
        Text(String(format: format, value))
          .font(.subheadline.monospacedDigit())
          .foregroundStyle(.secondary)
      }

      if showLabels && leftLabel != nil && rightLabel != nil {
        Slider(value: $value, in: range, step: step)
        HStack {
          Text(leftLabel!)
            .font(.caption2)
            .foregroundStyle(.secondary)
          Spacer()
          Text(rightLabel!)
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
      } else {
        Slider(value: $value, in: range, step: step)
      }

      if let description {
        Text(description)
          .font(.caption)
          .foregroundStyle(.tertiary)
      }
    }
  }
}

struct ParameterIntSlider: View {
  let title: String
  @Binding var value: Int
  var range: ClosedRange<Int> = 1...100
  var step: Int = 1
  var description: String? = nil

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      HStack {
        Text(title)
          .font(.subheadline.bold())
        Spacer()
        Text("\(value)")
          .font(.subheadline.monospacedDigit())
          .foregroundStyle(.secondary)
      }

      Slider(
        value: Binding(
          get: { Double(value) },
          set: { value = Int($0) }
        ),
        in: Double(range.lowerBound)...Double(range.upperBound),
        step: Double(step)
      )

      if let description {
        Text(description)
          .font(.caption)
          .foregroundStyle(.tertiary)
      }
    }
  }
}

struct SeedInputView: View {
  @Binding var seed: UInt64?
  @Binding var useRandomSeed: Bool

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Text("Randomize")
          .font(.subheadline.bold())
        Spacer()
        Toggle("", isOn: $useRandomSeed)
          .toggleStyle(.switch)
          .controlSize(.small)
      }

      if !useRandomSeed {
        HStack {
          TextField(
            "Seed value",
            value: Binding(
              get: { seed ?? 0 },
              set: { seed = $0 }
            ),
            format: .number
          )
          .textFieldStyle(.roundedBorder)
          .frame(maxWidth: 150)

          Button {
            seed = UInt64.random(in: 0...UInt64.max)
          } label: {
            Image(systemName: "dice")
          }
          .help("Generate random seed")
        }
      }
    }
  }
}

struct ResolutionPicker: View {
  @Binding var resolution: Int
  var options: [Int] = [640, 1024]

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Text("Quality")
          .font(.subheadline.bold())
        Spacer()
      }

      Picker("Quality", selection: $resolution) {
        Text("Standard").tag(640)
        Text("High").tag(1024)
      }
      .pickerStyle(.segmented)
    }
  }
}

struct LayersStepper: View {
  @Binding var layers: Int
  var range: ClosedRange<Int> = 1...8

  var body: some View {
    HStack {
      Text("Layers")
        .font(.subheadline.bold())
      Spacer()
      Stepper("\(layers)", value: $layers, in: range)
        .frame(width: 100)
    }
  }
}

struct CollapsibleSection<Content: View>: View {
  let title: String
  @Binding var isExpanded: Bool
  @ViewBuilder let content: () -> Content

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Button {
        withAnimation(.spring(duration: 0.25)) {
          isExpanded.toggle()
        }
      } label: {
        HStack {
          Text(title)
            .font(.subheadline.bold())
            .foregroundStyle(.primary)
          Spacer()
          Image(systemName: "chevron.right")
            .font(.caption)
            .foregroundStyle(.secondary)
            .rotationEffect(.degrees(isExpanded ? 90 : 0))
        }
      }
      .buttonStyle(.plain)

      if isExpanded {
        content()
          .transition(.opacity.combined(with: .move(edge: .top)))
      }
    }
  }
}

#Preview {
  ParameterControlsPreview()
}

private struct ParameterControlsPreview: View {
  @State private var floatValue: Float = 4.0
  @State private var intValue: Int = 50
  @State private var seed: UInt64? = nil
  @State private var useRandom = true
  @State private var resolution = 640
  @State private var layers = 4
  @State private var expanded = true

  var body: some View {
    Form {
      ParameterSlider(
        title: "CFG Scale",
        value: $floatValue,
        range: 1...10,
        step: 0.5,
        description: "Higher values follow prompts more closely"
      )

      ParameterIntSlider(
        title: "Steps",
        value: $intValue,
        range: 1...50,
        step: 1,
        description: "More steps = higher quality but slower"
      )

      SeedInputView(seed: $seed, useRandomSeed: $useRandom)
      ResolutionPicker(resolution: $resolution)
      LayersStepper(layers: $layers)

      CollapsibleSection(title: "Advanced Options", isExpanded: $expanded) {
        Text("Advanced content here")
          .foregroundStyle(.secondary)
      }
    }
    .formStyle(.grouped)
    .frame(width: 300)
    .padding()
  }
}
