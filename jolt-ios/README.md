# jolt-ios

C-compatible FFI layer for embedding jolt-core in iOS applications.

## Status

**Currently a skeleton/stub implementation.** The FFI functions compile but return "not implemented" errors. This crate establishes the API surface and build infrastructure.

## Architecture

This crate provides a minimal C API for:
- Generating proofs from RISC-V ELF binaries
- Verifying proofs
- Memory management for allocated buffers

### Current Implementation

The crate compiles with `jolt-core`'s `prover` feature but does not yet implement the actual proof generation and verification logic. Functions return `ErrorNotImplemented` status codes.

### Why Not Fully Implemented?

Jolt's high-level API (`Program`, preprocessing helpers, trace generation) is gated behind the `host` feature, which pulls in dependencies not suitable for iOS:
- `reqwest` (networking)
- `tokio` (async runtime)
- Filesystem operations expecting desktop paths (`~/.jolt`)
- Shell command execution

## Implementation Options

### Option A: Enable `host` Feature (easier, less portable)

1. Add `host` feature to the `jolt-core` dependency
2. Verify all transitive dependencies work on iOS
3. Mock or disable filesystem/network operations
4. Implement FFI functions using `Program::decode`, `program.preprocess()`, `JoltCpuProver::prove`

**Pros:** Uses the high-level API, less code to write  
**Cons:** Larger binary, pulls in dependencies that may not be iOS-compatible

### Option B: Low-Level API (harder, more portable)

1. Manually parse ELF binaries or accept pre-parsed bytecode
2. Use the `tracer` crate directly to generate execution traces
3. Call lower-level `jolt-core` functions for preprocessing
4. Construct `JoltCpuProver` and `JoltVerifier` manually
5. Serialize/deserialize proof data

**Pros:** Minimal dependencies, full control, better for embedded/constrained environments  
**Cons:** More code, tighter coupling to jolt-core internals

## Building

### For iOS Device (ARM64)

```bash
cd jolt-ios
cargo build --release --target aarch64-apple-ios --no-default-features
```

### For iOS Simulator (Apple Silicon)

```bash
cargo build --release --target aarch64-apple-ios-sim --no-default-features
```

### For iOS Simulator (Intel)

```bash
cargo build --release --target x86_64-apple-ios --no-default-features
```

### Using the Provided Script

From the repository root:

```bash
# Build all iOS targets and create .xcframework
JOLT_IOS_FEATURES="prover" \
JOLT_IOS_TARGETS="aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios" \
JOLT_IOS_HEADERS="$(pwd)/jolt-ios/include" \
./scripts/build-ios.sh
```

## Generating C Headers

Install `cbindgen`:

```bash
cargo install cbindgen
```

Generate headers:

```bash
cd jolt-ios
mkdir -p include
cbindgen --config cbindgen.toml --output include/jolt_ffi.h
```

## API Reference

### Functions

```c
// Get library version
const char* jolt_version(void);

// Get build information
const char* jolt_build_info(void);

// Generate a proof (NOT YET IMPLEMENTED)
jolt_proof_result jolt_prove(
    const uint8_t* elf_bytes,
    size_t elf_len,
    const uint8_t* input_bytes,
    size_t input_len,
    size_t max_input_size,
    size_t max_output_size
);

// Verify a proof (NOT YET IMPLEMENTED)
jolt_verify_result jolt_verify(
    const uint8_t* proof_data,
    size_t proof_len,
    const uint8_t* preprocessing_data,
    size_t preprocessing_len,
    const uint8_t* expected_output,
    size_t expected_output_len
);

// Free buffers allocated by this library
void jolt_free_buffer(uint8_t* ptr, size_t len);
```

### Status Codes

```c
typedef enum {
    SUCCESS = 0,
    ERROR_INVALID_INPUT = 1,
    ERROR_PROOF_GENERATION = 2,
    ERROR_VERIFICATION = 3,
    ERROR_SERIALIZATION = 4,
    ERROR_OUT_OF_MEMORY = 5,
    ERROR_NOT_IMPLEMENTED = 99,
} jolt_status_code;
```

## Next Steps

1. **Choose implementation approach** (Option A or B above)
2. **Implement `jolt_prove`**:
   - Parse ELF or accept preprocessed program
   - Generate execution trace
   - Run prover
   - Serialize proof
3. **Implement `jolt_verify`**:
   - Deserialize preprocessing and proof
   - Run verifier
   - Return validation result
4. **Add tests** using known-good ELF binaries
5. **Create example iOS app** demonstrating integration
6. **Performance tuning** for mobile hardware

## Testing

```bash
cargo test --no-default-features
```

Currently only tests the working functions (version info, memory management).

## Integration with Xcode

Once the FFI functions are implemented:

1. Generate headers: `cbindgen --output include/jolt_ffi.h`
2. Build for all targets: `JOLT_IOS_HEADERS=include ../scripts/build-ios.sh`
3. Drag `target/ios/JoltCore.xcframework` into your Xcode project
4. Import the header in your bridging header (Swift) or directly (Objective-C)
5. Call functions from your app

## License

Follows the jolt-core licensing (see repository root).

