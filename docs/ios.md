# Running `jolt-core` on iOS (no SSH)

Jolt's CLI depends on desktop-only tooling (`rustup`, `tar`, filesystem writes under `~/.jolt`, etc.). To run proofs directly on an iPhone you must embed the `jolt-core` library inside an Xcode app, disable the `host` feature set, and expose the prover/verifier logic through your own C-compatible API. This guide walks through the minimum viable setup.

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

- The default feature flag `host` pulls in blocking dependencies and is turned **off** for iOS builds by passing `--no-default-features --features "minimal prover"`.
- `Cargo.toml` now advertises a `staticlib` crate type so that Cargo emits `libjolt_core.a`, which can be linked into Swift/Objective-C projects.

## 3. Cross-compile the Rust static libraries

From the workspace root:

```bash
scripts/build-ios.sh
```

Environment knobs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `JOLT_IOS_FEATURES` | `"minimal prover"` | Feature set passed to `cargo`. Include extra features if your FFI shim needs them. |
| `JOLT_IOS_TARGETS` | `"aarch64-apple-ios x86_64-apple-ios"` | Space-separated list of targets to build. Remove `x86_64` if you only care about real devices (Apple Silicon simulators use `arm64`, set `JOLT_IOS_TARGETS="aarch64-apple-ios aarch64-apple-ios-sim"`). |
| `JOLT_IOS_BUILD_TYPE` | `release` | Use `debug` or the name of a custom Cargo profile if desired. |
| `JOLT_IOS_HEADERS` | _(unset)_ | Path to the directory that contains the headers generated for your FFI shim. When set, the script will call `xcodebuild -create-xcframework` to emit `target/ios/JoltCore.xcframework`. |

Artifacts are copied to `target/ios/libjolt_core-<target>.a`. You can manually run `xcodebuild -create-xcframework` later once you have headers.

## 4. Add a C-compatible facade

Create (or reuse) a small Rust crate that depends on `jolt-core`, exposes the functions you need via `#[no_mangle] extern "C"`, and ensures the inputs/outputs are plain-old-data. Example skeleton:

```rust
// ffi/src/lib.rs
use jolt_core::zkvm::{prover::JoltCpuProver, verifier::JoltVerifier};

#[repr(C)]
pub struct JoltStatus {
    pub ok: bool,
    pub cycles: u64,
}

#[no_mangle]
pub extern "C" fn jolt_prove(program: *const u8, program_len: usize) -> JoltStatus {
    // Safety + slice conversion omitted for brevity.
    // Instantiate preprocessing, run prover, serialize proof, etc.
    JoltStatus { ok: true, cycles: 0 }
}
```

- Build this crate with the same feature set as `jolt-core` (typically `minimal prover`).
- Run `cbindgen --config cbindgen.toml --crate ffi --output ios/include/jolt_ffi.h` to generate the header referenced by `JOLT_IOS_HEADERS`.
- Re-run `scripts/build-ios.sh` so the `.xcframework` includes your new header.

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
