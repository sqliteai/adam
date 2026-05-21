// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "adam",
    platforms: [.macOS(.v11), .iOS(.v14)],
    products: [
        .library(
            name: "adam",
            targets: ["adam"])
    ],
    targets: [
        .binaryTarget(
            name: "adamBinary",
            // The url + checksum below are auto-updated by .github/workflows/main.yml
            // on every release (see the "update Package.swift checksum and version"
            // step). The placeholder values let `swift package describe` succeed
            // before the first release; consumers should pin to a tagged version.
            url: "https://github.com/sqliteai/adam/releases/download/0.7.0/adam-apple-xcframework-0.7.0.zip",
            checksum: "ef1a4755901af4310e972b00633537eaa7b585b4e1c02493f0f6d6da4dbfbd8c"
        ),
        .target(
            name: "adam",
            dependencies: ["adamBinary"],
            path: "packages/swift"
        ),
    ]
)
