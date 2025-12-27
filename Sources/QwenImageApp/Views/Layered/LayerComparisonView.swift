import SwiftUI

/// Side-by-side comparison of original image with layers
struct LayerComparisonView: View {
  let layers: [NSImage]
  let inputImage: NSImage

  @Environment(\.dismiss) private var dismiss
  @State private var selectedLayerIndex = 0
  @State private var showOriginal = true
  @State private var opacity: Double = 1.0

  var body: some View {
    VStack(spacing: 0) {
      // Header
      headerView

      // Main comparison area
      comparisonArea

      // Layer selector
      layerSelector
    }
    .frame(minWidth: 800, minHeight: 600)
    .background(Color(nsColor: .windowBackgroundColor))
  }

  // MARK: - Header

  private var headerView: some View {
    HStack {
      Text("Compare Layers")
        .font(.headline)

      Spacer()

      Toggle("Show Original", isOn: $showOriginal)
        .toggleStyle(.switch)
        .controlSize(.small)

      if showOriginal {
        Slider(value: $opacity, in: 0...1)
          .frame(width: 150)
      }

      Button {
        dismiss()
      } label: {
        Image(systemName: "xmark.circle.fill")
          .foregroundStyle(.secondary)
      }
      .buttonStyle(.plain)
    }
    .padding()
    .background(Color(nsColor: .controlBackgroundColor))
  }

  // MARK: - Comparison Area

  private var comparisonArea: some View {
    GeometryReader { geometry in
      HStack(spacing: 20) {
        // Original image
        if showOriginal {
          originalImageView
            .frame(width: geometry.size.width / 2)
        }

        // Selected layer
        layerImageView
          .frame(maxWidth: .infinity)
      }
      .padding()
    }
  }

  private var originalImageView: some View {
    VStack(spacing: 8) {
      Text("Original")
        .font(.caption)
        .foregroundStyle(.secondary)

      Image(nsImage: inputImage)
        .resizable()
        .aspectRatio(contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
  }

  private var layerImageView: some View {
    VStack(spacing: 8) {
      HStack {
        Text("Layer \(selectedLayerIndex + 1)")
          .font(.caption)
          .foregroundStyle(.secondary)
        Spacer()
        Text("\(layers.count) total layers")
          .font(.caption)
          .foregroundStyle(.tertiary)
      }

      if selectedLayerIndex < layers.count {
        Image(nsImage: layers[selectedLayerIndex])
          .resizable()
          .aspectRatio(contentMode: .fit)
          .clipShape(RoundedRectangle(cornerRadius: 8))
          .overlay(
            RoundedRectangle(cornerRadius: 8)
              .strokeBorder(Color.accentColor, lineWidth: 2)
          )
      }
    }
  }

  // MARK: - Layer Selector

  private var layerSelector: some View {
    ScrollView(.horizontal, showsIndicators: false) {
      HStack(spacing: 12) {
        // Original thumbnail
        ThumbnailButton(
          image: inputImage,
          label: "Original",
          isSelected: false
        ) {
          // Can't select original, just show it in comparison
        }
        .opacity(showOriginal ? 1 : 0.3)

        Divider()
          .frame(height: 40)

        // Layer thumbnails
        ForEach(Array(layers.enumerated()), id: \.offset) { index, layer in
          ThumbnailButton(
            image: layer,
            label: "Layer \(index + 1)",
            isSelected: selectedLayerIndex == index
          ) {
            selectedLayerIndex = index
          }
        }
      }
      .padding(.horizontal)
      .padding(.vertical, 12)
    }
    .background(Color(nsColor: .controlBackgroundColor))
  }
}

// MARK: - Thumbnail Button

struct ThumbnailButton: View {
  let image: NSImage
  let label: String
  let isSelected: Bool
  let action: () -> Void

  var body: some View {
    Button(action: action) {
      VStack(spacing: 4) {
        Image(nsImage: image)
          .resizable()
          .aspectRatio(contentMode: .fill)
          .frame(width: 60, height: 45)
          .clipShape(RoundedRectangle(cornerRadius: 4))
          .overlay(
            RoundedRectangle(cornerRadius: 4)
              .strokeBorder(
                isSelected ? Color.accentColor : Color.secondary.opacity(0.3),
                lineWidth: isSelected ? 2 : 1
              )
          )

        Text(label)
          .font(.caption2)
          .foregroundStyle(isSelected ? .primary : .secondary)
      }
    }
    .buttonStyle(.plain)
  }
}

#Preview {
  LayerComparisonView(
    layers: [],
    inputImage: NSImage()
  )
}
