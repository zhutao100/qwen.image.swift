import SwiftUI
import UniformTypeIdentifiers

struct ImageDropZone: View {
  @Binding var droppedImage: NSImage?
  var placeholder: String = "Drop image here or click to browse"
  var onImageLoaded: ((URL) -> Void)?

  @State private var isTargeted = false
  @State private var showFileImporter = false

  var body: some View {
    ZStack {
      RoundedRectangle(cornerRadius: 12)
        .strokeBorder(
          isTargeted ? Color.accentColor : Color.secondary.opacity(0.3),
          style: StrokeStyle(lineWidth: 2, dash: droppedImage == nil ? [8] : [])
        )
        .background(
          RoundedRectangle(cornerRadius: 12)
            .fill(isTargeted ? Color.accentColor.opacity(0.1) : Color.secondary.opacity(0.03))
        )

      if let image = droppedImage {
        Image(nsImage: image)
          .resizable()
          .aspectRatio(contentMode: .fit)
          .padding(8)
          .overlay(alignment: .topTrailing) {
            Button {
              droppedImage = nil
            } label: {
              Image(systemName: "xmark.circle.fill")
                .font(.title2)
                .foregroundStyle(.white)
                .shadow(radius: 2)
            }
            .buttonStyle(.plain)
            .padding(12)
          }
      } else {
        VStack(spacing: 12) {
          Image(systemName: "photo.on.rectangle.angled")
            .font(.system(size: 48))
            .foregroundStyle(.secondary)

          Text(placeholder)
            .font(.headline)
            .foregroundStyle(.secondary)

          Text("PNG, JPEG, TIFF, or WebP")
            .font(.caption)
            .foregroundStyle(.tertiary)
        }
      }
    }
    .contentShape(Rectangle())
    .onDrop(of: [.image, .fileURL, .webp], isTargeted: $isTargeted) { providers in
      handleDrop(providers: providers)
    }
    .onTapGesture {
      if droppedImage == nil {
        showFileImporter = true
      }
    }
    .fileImporter(
      isPresented: $showFileImporter,
      allowedContentTypes: [.image, .webp],
      allowsMultipleSelection: false
    ) { result in
      handleFileImport(result)
    }
  }

  private func handleDrop(providers: [NSItemProvider]) -> Bool {
    guard let provider = providers.first else { return false }

    if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
      provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
        guard let data = item as? Data,
              let url = URL(dataRepresentation: data, relativeTo: nil)
        else { return }

        Task { @MainActor in
          loadImage(from: url)
        }
      }
      return true
    }

    if provider.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
      provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
        guard let data = data,
              let image = NSImage(data: data)
        else { return }

        DispatchQueue.main.async {
          droppedImage = image
        }
      }
      return true
    }

    return false
  }

  private func handleFileImport(_ result: Result<[URL], Error>) {
    switch result {
    case .success(let urls):
      if let url = urls.first {
        loadImage(from: url)
      }
    case .failure(let error):
      print("File import error: \(error.localizedDescription)")
    }
  }

  private func loadImage(from url: URL) {
    do {
      let image = try ImageIOService.loadImage(from: url)
      droppedImage = image
      onImageLoaded?(url)
    } catch {
      print("Failed to load image: \(error.localizedDescription)")
    }
  }
}

#Preview {
  ImageDropZonePreview()
}

private struct ImageDropZonePreview: View {
  @State private var image: NSImage? = nil

  var body: some View {
    ImageDropZone(droppedImage: $image)
      .frame(width: 400, height: 300)
      .padding()
  }
}
