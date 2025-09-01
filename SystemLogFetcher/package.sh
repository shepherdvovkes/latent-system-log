#!/bin/bash

# System Log Fetcher Packaging Script
echo "Packaging System Log Fetcher..."

# Configuration
APP_NAME="SystemLogFetcher"
BUNDLE_ID="com.systematiclabs.systemlogfetcher"
VERSION="2.0.0"
BUILD_DIR="build"
APP_BUNDLE_DIR="$BUILD_DIR/$APP_NAME.app"
CONTENTS_DIR="$APP_BUNDLE_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

# Clean previous builds
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Create app bundle structure
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Build the executable
echo "Building executable..."
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
    -o "$MACOS_DIR/$APP_NAME"

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Make executable
chmod +x "$MACOS_DIR/$APP_NAME"

# Create Info.plist
cat > "$CONTENTS_DIR/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.utilities</string>
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeName</key>
            <string>System Log File</string>
            <key>CFBundleTypeExtensions</key>
            <array>
                <string>log</string>
                <string>json</string>
                <string>csv</string>
            </array>
            <key>CFBundleTypeRole</key>
            <string>Viewer</string>
        </dict>
    </array>
</dict>
</plist>
EOF

# Create PkgInfo
echo "APPL????" > "$CONTENTS_DIR/PkgInfo"

# Create app icon placeholder (you can replace this with a real icon)
mkdir -p "$RESOURCES_DIR"
touch "$RESOURCES_DIR/AppIcon.icns"

# Create entitlements for system log access
cat > "$BUILD_DIR/entitlements.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.app-sandbox</key>
    <true/>
    <key>com.apple.security.files.user-selected.read-write</key>
    <true/>
    <key>com.apple.security.network.client</key>
    <true/>
    <key>com.apple.security.device.usb</key>
    <true/>
    <key>com.apple.security.automation.apple-events</key>
    <true/>
</dict>
</plist>
EOF

# Sign the app (optional - requires developer certificate)
if command -v codesign &> /dev/null; then
    echo "Signing app bundle..."
    codesign --force --deep --sign - "$APP_BUNDLE_DIR"
fi

# Create DMG (optional)
if command -v create-dmg &> /dev/null; then
    echo "Creating DMG..."
    create-dmg \
        --volname "$APP_NAME" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "$APP_NAME.app" 175 120 \
        --hide-extension "$APP_NAME.app" \
        --app-drop-link 425 120 \
        "$BUILD_DIR/$APP_NAME-$VERSION.dmg" \
        "$APP_BUNDLE_DIR"
else
    echo "create-dmg not found. Skipping DMG creation."
fi

# Create zip archive
echo "Creating zip archive..."
cd "$BUILD_DIR"
zip -r "$APP_NAME-$VERSION.zip" "$APP_NAME.app"
cd ..

echo "Packaging completed successfully!"
echo "App bundle: $APP_BUNDLE_DIR"
echo "Zip archive: $BUILD_DIR/$APP_NAME-$VERSION.zip"
if [ -f "$BUILD_DIR/$APP_NAME-$VERSION.dmg" ]; then
    echo "DMG: $BUILD_DIR/$APP_NAME-$VERSION.dmg"
fi
echo ""
echo "To run the app: open $APP_BUNDLE_DIR"
echo "To install: drag $APP_NAME.app to Applications folder"
