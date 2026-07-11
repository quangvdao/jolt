# iOS Integration Guide

This document explains how to integrate the Jolt proving system into an iOS app using the pre-built `JoltCore.xcframework`.

## Quick Start

### 1. Build the Framework

From the repository root:

```bash
JOLT_IOS_HEADERS="$(pwd)/jolt-ios/include" ./scripts/build-ios.sh
```

This creates:
- `target/ios/JoltCore.xcframework` - Ready to drop into Xcode
- Individual `.a` files for each architecture in `target/ios/`

### 2. Add to Xcode Project

1. Open your iOS project in Xcode
2. Drag `target/ios/JoltCore.xcframework` into your project navigator
3. Ensure "Copy items if needed" is checked
4. Add to your target's "Frameworks, Libraries, and Embedded Content"
5. Set "Embed" to "Do Not Embed" (it's a static library)

### 3. Import and Use

#### Swift (via Bridging Header)

Create or update your bridging header:

```objective-c
// YourProject-Bridging-Header.h
#import <JoltCore/jolt_ffi.h>
```

Use in Swift:

```swift
import Foundation

class JoltProver {
    func getVersion() -> String {
        guard let cString = jolt_version() else {
            return "Unknown"
        }
        return String(cString: cString)
    }
    
    func getBuildInfo() -> String {
        guard let cString = jolt_build_info() else {
            return "Unknown"
        }
        return String(cString: cString)
    }
    
    // Proof generation (currently returns ERROR_NOT_IMPLEMENTED)
    func prove(elfData: Data, input: Data) -> Result<ProofData, JoltError> {
        return elfData.withUnsafeBytes { elfBytes in
            input.withUnsafeBytes { inputBytes in
                let result = jolt_prove(
                    elfBytes.baseAddress?.assumingMemoryBound(to: UInt8.self),
                    elfData.count,
                    inputBytes.baseAddress?.assumingMemoryBound(to: UInt8.self),
                    input.count,
                    1024 * 1024,  // max_input_size
                    1024 * 1024   // max_output_size
                )
                
                if result.status == SUCCESS {
                    let data = Data(bytes: result.proof_data, count: result.proof_len)
                    // Don't forget to free!
                    jolt_free_buffer(result.proof_data, result.proof_len)
                    return .success(ProofData(data: data, cycles: result.cycles))
                } else {
                    return .failure(JoltError(code: result.status))
                }
            }
        }
    }
}

struct ProofData {
    let data: Data
    let cycles: UInt64
}

enum JoltError: Error {
    case notImplemented
    case invalidInput
    case proofGeneration
    case verification
    case serialization
    case outOfMemory
    
    init(code: jolt_status_code) {
        switch code {
        case ERROR_NOT_IMPLEMENTED:
            self = .notImplemented
        case ERROR_INVALID_INPUT:
            self = .invalidInput
        case ERROR_PROOF_GENERATION:
            self = .proofGeneration
        case ERROR_VERIFICATION:
            self = .verification
        case ERROR_SERIALIZATION:
            self = .serialization
        case ERROR_OUT_OF_MEMORY:
            self = .outOfMemory
        default:
            self = .notImplemented
        }
    }
}
```

#### Objective-C

```objective-c
#import <JoltCore/jolt_ffi.h>

- (void)checkJoltVersion {
    const char *version = jolt_version();
    NSLog(@"Jolt version: %s", version);
    
    const char *buildInfo = jolt_build_info();
    NSLog(@"Build info: %s", buildInfo);
}
```

## Current Limitations

**The FFI functions are currently stubs.** They compile and link correctly, but:

- `jolt_prove()` returns `ERROR_NOT_IMPLEMENTED`
- `jolt_verify()` returns `ERROR_NOT_IMPLEMENTED`
- Only `jolt_version()`, `jolt_build_info()`, and `jolt_free_buffer()` work

See `jolt-ios/README.md` for implementation roadmap.

## Architecture Support

The `JoltCore.xcframework` includes:

- **iOS Device** (aarch64-apple-ios): For physical iPhones/iPads
- **iOS Simulator** (universal): Combined aarch64 (Apple Silicon Macs) and x86_64 (Intel Macs)

## Memory Management

All buffers returned by Jolt functions must be freed:

```swift
let result = jolt_prove(...)
if result.status == SUCCESS {
    // Use result.proof_data...
    
    // Always free when done!
    jolt_free_buffer(result.proof_data, result.proof_len)
}
```

## Threading

Once the prover is implemented, proof generation will be CPU-intensive. Run on a background thread:

```swift
Task.detached(priority: .userInitiated) {
    let result = prover.prove(elf: elfData, input: inputData)
    
    await MainActor.run {
        // Update UI with result
    }
}
```

Consider using `ProcessInfo.processInfo.activeProcessorCount` to tune parallelism via the `RAYON_NUM_THREADS` environment variable:

```swift
setenv("RAYON_NUM_THREADS", "4", 1)
```

## Build Script Options

Customize the build by setting environment variables:

```bash
# Build only for device (no simulator)
JOLT_IOS_TARGETS="aarch64-apple-ios" \
JOLT_IOS_HEADERS="$(pwd)/jolt-ios/include" \
./scripts/build-ios.sh

# Debug build
JOLT_IOS_BUILD_TYPE="debug" \
JOLT_IOS_HEADERS="$(pwd)/jolt-ios/include" \
./scripts/build-ios.sh

# Build jolt-core directly (for advanced use cases)
JOLT_IOS_PACKAGE="jolt-core" \
JOLT_IOS_FEATURES="minimal prover" \
./scripts/build-ios.sh
```

## File Sizes

The static libraries are large due to cryptographic operations:

- `jolt-ios`: ~5 MB per architecture
- `jolt-core`: ~385 MB per architecture (includes all Jolt internals)

The `jolt-ios` FFI layer is much smaller because it only includes the symbols it exports.

## Troubleshooting

### "Framework not found JoltCore"

Ensure the xcframework is in your project and added to your target's "Frameworks, Libraries, and Embedded Content".

### Linker errors about missing symbols

The xcframework might not be embedded correctly. Check:
1. It's in "Frameworks, Libraries, and Embedded Content"
2. "Embed" is set to "Do Not Embed" (static framework)

### Runtime error: "Symbol not found"

Check that your bridging header imports the correct path:

```objective-c
#import <JoltCore/jolt_ffi.h>  // Correct
// NOT: #import "jolt_ffi.h"
```

### Build errors after updating Rust code

Rebuild the xcframework:

```bash
cd jolt-ios
cbindgen --config cbindgen.toml --output include/jolt_ffi.h
cd ..
JOLT_IOS_HEADERS="$(pwd)/jolt-ios/include" ./scripts/build-ios.sh
```

Then replace the xcframework in your Xcode project (delete the old one first).

## Next Steps

To implement actual proof generation/verification:

1. Review `jolt-ios/README.md` for implementation options
2. Choose between high-level API (Option A) or low-level API (Option B)
3. Implement the FFI functions in `jolt-ios/src/lib.rs`
4. Rebuild and test with real ELF binaries
5. Profile and optimize for mobile hardware

## Example Project

A complete example iOS app demonstrating integration will be added in a future update.

