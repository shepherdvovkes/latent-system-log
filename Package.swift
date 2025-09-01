// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SystemLogFetcher",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .executable(
            name: "SystemLogFetcher",
            targets: ["SystemLogFetcher"]
        ),
    ],
    dependencies: [
        // No external dependencies needed for this project
    ],
    targets: [
        .executableTarget(
            name: "SystemLogFetcher",
            dependencies: [],
            path: "Sources"
        ),
    ]
)
