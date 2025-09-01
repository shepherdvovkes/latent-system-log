#!/bin/bash

# System Log Fetcher Build Script
echo "Building System Log Fetcher..."

# Create build directory
mkdir -p build

# Compile Swift files
swiftc \
    -target x86_64-apple-macosx14.0 \
    -sdk $(xcrun --show-sdk-path) \
    -I $(xcrun --show-sdk-path)/System/Library/Frameworks \
    -framework Foundation \
    -framework SwiftUI \
    -framework AppKit \
    -framework OSLog \
    ../Sources/SystemLogFetcherApp.swift \
    ../Sources/ContentView.swift \
    ../Sources/LogFetcher.swift \
    ../Sources/DatabaseManager.swift \
    -o build/SystemLogFetcher

if [ $? -eq 0 ]; then
    echo "Build successful! Executable created at build/SystemLogFetcher"
    echo "To run the application: ./build/SystemLogFetcher"
else
    echo "Build failed!"
    exit 1
fi
