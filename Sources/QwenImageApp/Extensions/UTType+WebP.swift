import UniformTypeIdentifiers

extension UTType {
  static var webp: UTType {
    UTType(filenameExtension: "webp") ?? .image
  }
}
