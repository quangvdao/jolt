# jolt-android

JNI (Java Native Interface) layer for embedding jolt-core in Android applications.

## Status

**Currently a skeleton/stub implementation.** The JNI functions compile but return "not implemented" errors. This crate establishes the API surface and build infrastructure.

## Architecture

This crate provides a JNI API for:
- Generating proofs from RISC-V ELF binaries
- Verifying proofs
- Accessing library metadata

### Current Implementation

The crate compiles with `jolt-core`'s `prover` feature but does not yet implement the actual proof generation and verification logic. Functions return `ERROR_NOT_IMPLEMENTED` status codes.

### Why Not Fully Implemented?

Jolt's high-level API (`Program`, preprocessing helpers, trace generation) is gated behind the `host` feature, which pulls in dependencies not suitable for Android:
- `reqwest` (networking) - works on Android but needs configuration
- `tokio` (async runtime) - works on Android
- Filesystem operations expecting desktop paths (`~/.jolt`)
- Shell command execution

## Implementation Options

### Option A: Enable `host` Feature (easier, less portable)

1. Add `host` feature to the `jolt-core` dependency
2. Verify all transitive dependencies work on Android
3. Replace filesystem paths with Android-appropriate locations (use Context.getFilesDir())
4. Implement JNI functions using `Program::decode`, `program.preprocess()`, `JoltCpuProver::prove`

**Pros:** Uses the high-level API, less code to write  
**Cons:** Larger binary, pulls in dependencies that may need Android-specific configuration

### Option B: Low-Level API (harder, more portable)

1. Manually parse ELF binaries or accept pre-parsed bytecode
2. Use the `tracer` crate directly to generate execution traces
3. Call lower-level `jolt-core` functions for preprocessing
4. Construct `JoltCpuProver` and `JoltVerifier` manually
5. Serialize/deserialize proof data

**Pros:** Minimal dependencies, full control, better for embedded/constrained environments  
**Cons:** More code, tighter coupling to jolt-core internals

## Prerequisites

### 1. Install Android Targets

```bash
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi
rustup target add x86_64-linux-android
rustup target add i686-linux-android
```

### 2. Install Android NDK

Install via Android Studio's SDK Manager or download directly:
- Open Android Studio → SDK Manager → SDK Tools → NDK (Side by side)
- Or download from: https://developer.android.com/ndk/downloads

### 3. Set Environment Variables

```bash
export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/26.1.10909125
# Adjust version number to match your installation
```

Add to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/26.1.10909125
export PATH=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin:$PATH
```

### 4. Configure Cargo for Android

Create or update `.cargo/config.toml` in the repository root:

```toml
[target.aarch64-linux-android]
ar = "aarch64-linux-android-ar"
linker = "aarch64-linux-android30-clang"

[target.armv7-linux-androideabi]
ar = "arm-linux-androideabi-ar"
linker = "armv7a-linux-androideabi30-clang"

[target.i686-linux-android]
ar = "i686-linux-android-ar"
linker = "i686-linux-android30-clang"

[target.x86_64-linux-android]
ar = "x86_64-linux-android-ar"
linker = "x86_64-linux-android30-clang"
```

The `30` in the linker names refers to Android API level 30. Adjust if you need a different minimum API level.

## Building

### Basic Build

From the repository root:

```bash
./scripts/build-android.sh
```

This builds for all Android architectures and creates the `jniLibs/` directory structure.

### With Java Wrapper Generation

```bash
JOLT_ANDROID_JAVA_OUT="$(pwd)/target/android/java" ./scripts/build-android.sh
```

### Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `JOLT_ANDROID_PACKAGE` | `"jolt-android"` | Cargo package name to build |
| `JOLT_ANDROID_FEATURES` | `""` | Feature set passed to `cargo` |
| `JOLT_ANDROID_TARGETS` | `"aarch64-linux-android armv7-linux-androideabi x86_64-linux-android i686-linux-android"` | Space-separated list of targets |
| `JOLT_ANDROID_BUILD_TYPE` | `release` | Use `debug` or custom profile name |
| `JOLT_ANDROID_JAVA_OUT` | _(unset)_ | Path to generate Java wrapper. When set, creates `JoltProver.java` |

### Build Artifacts

```
target/android/
├── jniLibs/
│   ├── arm64-v8a/
│   │   └── libjolt_android.so
│   ├── armeabi-v7a/
│   │   └── libjolt_android.so
│   ├── x86_64/
│   │   └── libjolt_android.so
│   └── x86/
│       └── libjolt_android.so
└── java/
    └── JoltProver.java  (if JOLT_ANDROID_JAVA_OUT is set)
```

## Integration with Android Project

### 1. Copy Libraries

```bash
cp -r target/android/jniLibs app/src/main/
```

Or configure Gradle to use the target directory directly:

```gradle
android {
    sourceSets {
        main {
            jniLibs.srcDirs = ['../../target/android/jniLibs']
        }
    }
}
```

### 2. Add Java Wrapper

Copy `target/android/java/JoltProver.java` to `app/src/main/java/com/jolt/`

Or add it to your package structure and update the package name.

### 3. Use in Kotlin/Java

```kotlin
import com.jolt.JoltProver

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Test JNI is working
        val version = JoltProver.getVersion()
        Log.d("Jolt", "Version: $version")
        
        val sum = JoltProver.testAdd(5, 7)
        Log.d("Jolt", "5 + 7 = $sum")  // Should print 12
        
        // Generate proof (currently returns not implemented)
        lifecycleScope.launch(Dispatchers.IO) {
            val elfBytes = loadElfFromAssets()
            val inputBytes = prepareInput()
            
            val result = JoltProver.prove(
                elfBytes,
                inputBytes,
                1024 * 1024,  // maxInputSize
                1024 * 1024   // maxOutputSize
            )
            
            when {
                result.isNotImplemented() -> {
                    Log.w("Jolt", "Proof generation not yet implemented")
                }
                result.isSuccess() -> {
                    Log.d("Jolt", "Proof generated: ${result.cycles} cycles")
                    // Use result.proofData
                }
                else -> {
                    Log.e("Jolt", "Proof generation failed: ${result.status}")
                }
            }
        }
    }
}
```

## API Reference

### Java/Kotlin API

```java
public class JoltProver {
    // Get library version
    public static native String getVersion();
    
    // Get build information
    public static native String getBuildInfo();
    
    // Generate a proof (NOT YET IMPLEMENTED)
    public static native ProofResult prove(
        byte[] elfBytes,
        byte[] inputBytes,
        int maxInputSize,
        int maxOutputSize
    );
    
    // Verify a proof (NOT YET IMPLEMENTED)
    public static native VerifyResult verify(
        byte[] proofData,
        byte[] preprocessingData,
        byte[] expectedOutput
    );
    
    // Test function
    public static native int testAdd(int a, int b);
    
    // Result classes
    public static class ProofResult {
        public final int status;
        public final long cycles;
        public final byte[] proofData;
    }
    
    public static class VerifyResult {
        public final int status;
        public final boolean isValid;
    }
    
    // Status codes
    public static class StatusCode {
        public static final int SUCCESS = 0;
        public static final int ERROR_INVALID_INPUT = 1;
        public static final int ERROR_PROOF_GENERATION = 2;
        public static final int ERROR_VERIFICATION = 3;
        public static final int ERROR_SERIALIZATION = 4;
        public static final int ERROR_OUT_OF_MEMORY = 5;
        public static final int ERROR_NOT_IMPLEMENTED = 99;
    }
}
```

## Threading and Performance

Proof generation will be CPU-intensive. Always run on a background thread:

```kotlin
// Kotlin Coroutines
lifecycleScope.launch(Dispatchers.IO) {
    val result = JoltProver.prove(...)
}

// Or with WorkManager for long-running tasks
class ProofWorker(context: Context, params: WorkerParameters) : 
    CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val proofResult = JoltProver.prove(...)
        // ...
        return Result.success()
    }
}
```

Tune parallelism based on device capabilities:

```kotlin
val numCores = Runtime.getRuntime().availableProcessors()
// Jolt uses Rayon internally, which respects RAYON_NUM_THREADS
```

## Testing

```bash
cargo test --package jolt-android
```

Currently only tests the working functions (status code values).

## Troubleshooting

### "UnsatisfiedLinkError: dlopen failed"

- Ensure `libjolt_android.so` is in the correct `jniLibs/` ABI directory
- Check that the library was built for the target device architecture
- Verify `System.loadLibrary("jolt_android")` uses the correct name (no "lib" prefix, no ".so" extension)

### "No implementation found for native method"

- Java class name must match: `com.jolt.JoltProver`
- If you changed the package, update the JNI function names in `src/lib.rs`:
  ```rust
  #[no_mangle]
  pub extern "C" fn Java_your_package_ClassName_methodName(...)
  ```

### "ANDROID_NDK_HOME not set"

Install Android NDK and export the environment variable:

```bash
export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/26.1.10909125
```

### Linker errors during build

Update `.cargo/config.toml` with correct NDK paths. The API level (e.g., `30` in `aarch64-linux-android30-clang`) should match your minimum supported Android version.

## Next Steps

To implement actual proof generation/verification:

1. Review implementation options (A or B above)
2. Choose approach based on your requirements
3. Implement the JNI functions in `src/lib.rs`:
   - Extract byte arrays from JNI
   - Call jolt-core APIs
   - Handle errors and return results
4. Test with real ELF binaries
5. Profile and optimize for mobile hardware

## File Sizes

Native libraries for Android:

- `jolt-android`: ~5-10 MB per architecture (depends on optimization)
- `jolt-core` (if building directly): ~300-400 MB per architecture

The `jolt-android` FFI layer is much smaller because it only includes exported symbols.

## License

Follows the jolt-core licensing (see repository root).

