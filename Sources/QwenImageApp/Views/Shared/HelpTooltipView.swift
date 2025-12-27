import SwiftUI

struct HelpTooltip: View {
    let title: String
    let content: String
    @State private var isShowingPopover = false

    var body: some View {
        Button {
            isShowingPopover = true
        } label: {
            Image(systemName: "questionmark.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .popover(isPresented: $isShowingPopover) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(title)
                        .font(.headline)
                    Spacer()
                    Button {
                        isShowingPopover = false
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                Divider()
                Text(content)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.leading)
            }
            .padding()
            .frame(minWidth: 250, maxWidth: 350)
        }
    }
}

struct LabeledParameterView: View {
    let title: String
    let tooltip: String
    let content: String
    @State private var isShowingPopover = false

    var body: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 4) {
                    Text(title)
                        .font(.subheadline.bold())
                    Button {
                        isShowingPopover = true
                    } label: {
                        Image(systemName: "questionmark.circle")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                Text(content)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            Spacer()
        }
        .popover(isPresented: $isShowingPopover) {
            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.headline)
                Text(tooltip)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .padding()
            .frame(minWidth: 200)
        }
    }
}

enum ParameterHelp {
    static let layers = """
    The number of layers to extract from your image.

    • 2-3 layers: Simple images with clear foreground/background
    • 4-6 layers: Complex images with multiple elements
    • 6+ layers: Very detailed images

    More layers take longer to generate.
    """

    static let quality = """
    The resolution of the output images.

    • Standard (640px): Faster, good for most uses
    • High (1024px): More detail, takes ~4x longer

    Choose Standard unless you need very high resolution output.
    """

    static let steps = """
    The number of processing steps the AI takes.

    • Fewer steps: Faster, may be less accurate
    • More steps: Slower, more refined results

    For most images, 50 steps provides good results.
    """

    static let trueCFGScale = """
    True Classifier-Free Guidance scale.

    Controls how strongly the model follows the conditioning.

    • Low (1-2): Minimal guidance, more variation
    • Medium (3-5): Balanced guidance
    • High (6+): Strong guidance, follows conditioning closely

    Default is 4.0 for most use cases.
    """

    static let prompt = """
    Describe what you want to extract or modify.

    Be specific about:
    • What objects should be in each layer
    • Background vs foreground elements
    • Any specific editing instructions

    Example: "Extract the cat as a transparent layer, keep the background"
    """

    static let negativePrompt = """
    Describe what you want to avoid in the output.

    Common examples:
    • "blurry, low quality" - for sharper results
    • "distorted, malformed" - for cleaner shapes
    • "dark, overexposed" - for proper lighting

    Leave empty if unsure.
    """

    static let seed = """
    A number that controls the randomness of generation.

    • Same image + same seed = same result
    • Different seed = different variation

    Use a fixed seed to reproduce a result you like.
    """
}

#Preview {
    VStack(spacing: 20) {
        HelpTooltip(
            title: "True CFG Scale",
            content: "Classifier-Free Guidance scale. Higher values follow prompts more closely."
        )

        LabeledParameterView(
            title: "Layers",
            tooltip: ParameterHelp.layers,
            content: "Number of layers to extract from your image"
        )
    }
    .padding()
}
