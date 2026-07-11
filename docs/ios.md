# Running `jolt-core` on iOS (no SSH)

Jolt's CLI depends on desktop-only tooling (`rustup`, `tar`, filesystem writes under `~/.jolt`, etc.). To run proofs directly on an iPhone, use the `jolt-ios` FFI crate which provides a C-compatible API for embedding Jolt in iOS apps. This guide walks through the setup and integration process.

**Note:** The `jolt-ios` crate currently provides a skeleton API. Full proof generation and verification requires additional implementation work (see `jolt-ios/README.md`).

## 1. Prerequisites

1. macOS host with Xcode + Command Line Tools installed.
2. Rust stable toolchain with iOS targets:
   ```bash
   rustup target add aarch64-apple-ios x86_64-apple-ios
   ```
3. Optional tooling:
   - `cbindgen` (to generate C headers from Rust FFI shims): `cargo install cbindgen`.
   - `xcodebuild` (bundled with Xcode) for producing `.xcframework`s.

## 2. Build configuration (already in repo)

- The `jolt-ios` crate provides a C FFI layer on top of `jolt-core`
- It uses the `prover` feature (which includes `minimal`) without the `host` feature
- `jolt-ios/Cargo.toml` configures `crate-type = ["staticlib", "cdylib"]` so Cargo emits `libjolt_ios.a`
- C headers are auto-generated using `cbindgen` from the Rust code

## 3. Cross-compile the Rust static libraries

From the workspace root:

```bash
scripts/build-ios.sh
```

Environment knobs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `JOLT_IOS_PACKAGE` | `"jolt-ios"` | Cargo package name to build. Use `"jolt-core"` if building the core library directly. |
| `JOLT_IOS_FEATURES` | `"prover"` | Feature set passed to `cargo`. |
| `JOLT_IOS_TARGETS` | `"aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios"` | Space-separated list of targets. Includes device (aarch64), Apple Silicon sim (aarch64-sim), and Intel sim (x86_64). |
| `JOLT_IOS_BUILD_TYPE` | `release` | Use `debug` or a custom Cargo profile name. |
| `JOLT_IOS_HEADERS` | _(unset)_ | Path to headers directory. When set, creates `target/ios/JoltCore.xcframework`. For `jolt-ios`, use `jolt-ios/include`. |

Artifacts are copied to `target/ios/libjolt_ios-<target>.a`. If `JOLT_IOS_HEADERS` is set, an `.xcframework` bundle is created for easy Xcode integration.

## 4. The `jolt-ios` FFI crate

The repository includes a `jolt-ios` crate that provides a C-compatible API:

```c
// Generate a proof from an ELF binary
jolt_proof_result jolt_prove(
    const uint8_t* elf_bytes,
    size_t elf_len,
    const uint8_t* input_bytes,
    size_t input_len,
    size_t max_input_size,
    size_t max_output_size
);

// Verify a proof
jolt_verify_result jolt_verify(
    const uint8_t* proof_data,
    size_t proof_len,
    const uint8_t* preprocessing_data,
    size_t preprocessing_len,
    const uint8_t* expected_output,
    size_t expected_output_len
);

// Memory management
void jolt_free_buffer(uint8_t* ptr, size_t len);
```

**Current Status:** The crate compiles and exports the API, but `jolt_prove` and `jolt_verify` return `ERROR_NOT_IMPLEMENTED`. See `jolt-ios/README.md` for implementation details and next steps.

To regenerate headers after modifying the FFI:

```bash
cd jolt-ios
cbindgen --config cbindgen.toml --output include/jolt_ffi.h
```

## 5. Integrate inside an Xcode app

1. Drag `target/ios/JoltCore.xcframework` (or the raw `.a` files) plus the generated headers into your Xcode project.
2. Ensure the consuming target links against `libc++`, `Accelerate`, or any other frameworks your app already uses—`jolt-core` itself only depends on `libc`.
3. Add the header to your bridging header (Swift) or import list (Objective-C). Call your exported functions just like any other C API.
4. Manage threading by setting `RAYON_NUM_THREADS` before the first prover invocation:
   ```swift
   setenv("RAYON_NUM_THREADS", "4", 1)
   ```
5. Handle long-running work via `BGProcessingTask` or `Task { @MainActor in ... }` to keep the UI responsive.

## 6. Runtime considerations

- Memory: complex proofs can easily exceed 2–3 GB. Monitor `ProcessInfo.processInfo.physicalMemory` and down-sample workloads when running on devices with 6 GB RAM or less.
- Filesystem: replace any hard-coded paths with directories returned by `FileManager.default.urls(for:in:)`. Keep artifacts inside your app group or temporary directory.
- Logging: pipe `tracing` output through `os_log` by setting `RUST_LOG=info` and capturing stdout/stderr with `OSLogStore` if needed.
- Updates: when upstream `jolt-core` changes, re-run the script to refresh the `.xcframework` and redistribute the app through TestFlight/App Store.

With these steps you have a fully local path—no SSH tunnel required—to drive Jolt proofs on iPhone hardware. Use the provided script + feature settings to rebuild whenever you change the FFI layer or upgrade the prover.
