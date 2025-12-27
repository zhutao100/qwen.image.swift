import SwiftUI

struct ProgressOverlay: View {
  let step: Int
  let total: Int
  let progress: Float
  var title: String = "Generating"
  var onCancel: (() -> Void)?

  var body: some View {
    ZStack {
      Color.black.opacity(0.5)
        .ignoresSafeArea()

      VStack(spacing: 20) {
        ZStack {
          Circle()
            .stroke(Color.secondary.opacity(0.3), lineWidth: 8)
            .frame(width: 80, height: 80)

          Circle()
            .trim(from: 0, to: CGFloat(progress))
            .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 8, lineCap: .round))
            .frame(width: 80, height: 80)
            .rotationEffect(.degrees(-90))
            .animation(.linear(duration: 0.3), value: progress)

          Text("\(Int(progress * 100))%")
            .font(.headline.monospacedDigit())
        }

        VStack(spacing: 8) {
          Text(title)
            .font(.title3.bold())

          Text("Step \(step) of \(total)")
            .font(.subheadline)
            .foregroundStyle(.secondary)

          if step > 0 {
            Text(estimatedTimeRemaining)
              .font(.caption)
              .foregroundStyle(.tertiary)
          }
        }

        if let onCancel {
          Button("Cancel", role: .destructive, action: onCancel)
            .keyboardShortcut(.escape, modifiers: [])
            .buttonStyle(.bordered)
        }
      }
      .padding(40)
      .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
      .shadow(radius: 20)
    }
  }

  private var estimatedTimeRemaining: String {
    guard progress > 0.01 else { return "Calculating..." }

    let remainingSteps = total - step
    let avgSecondsPerStep: Double = 8.5
    let remainingSeconds = Double(remainingSteps) * avgSecondsPerStep

    if remainingSeconds < 60 {
      return "About \(Int(remainingSeconds))s remaining"
    } else {
      let minutes = Int(remainingSeconds / 60)
      let seconds = Int(remainingSeconds.truncatingRemainder(dividingBy: 60))
      return "About \(minutes)m \(seconds)s remaining"
    }
  }
}

struct LoadingOverlay: View {
  var message: String = "Loading..."

  var body: some View {
    ZStack {
      Color.black.opacity(0.4)
        .ignoresSafeArea()

      VStack(spacing: 16) {
        ProgressView()
          .scaleEffect(1.5)
          .progressViewStyle(.circular)

        Text(message)
          .font(.headline)
      }
      .padding(40)
      .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
  }
}

#Preview("Progress Overlay") {
  ProgressOverlay(
    step: 15,
    total: 50,
    progress: 0.3,
    title: "Generating Layers",
    onCancel: {}
  )
}

#Preview("Loading Overlay") {
  LoadingOverlay(message: "Loading model...")
}
