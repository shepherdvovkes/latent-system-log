#!/bin/bash

echo "Building System Log Fetcher for macOS 15.0..."

# Clean previous build
rm -rf .build

# Build with specific target and ignore warnings
swift build \
    -Xswiftc -target \
    -Xswiftc arm64-apple-macosx15.0 \
    -Xswiftc -suppress-warnings

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Executable location: .build/debug/SystemLogFetcher"
    echo "To run: .build/debug/SystemLogFetcher"
else
    echo "Build failed!"
    exit 1
fi
