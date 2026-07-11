# Android Integration Guide

This document explains how to integrate the Jolt proving system into an Android app using the JNI wrapper.

## Prerequisites Checklist

Before building:

- [ ] Rust toolchain installed (`rustup` available)
- [ ] Android NDK installed (via Android Studio SDK Manager)
- [ ] Android targets added: `rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android i686-linux-android`
- [ ] `ANDROID_NDK_HOME` environment variable set
- [ ] `.cargo/config.toml` configured with NDK linkers

## Quick Start

### 1. Set Up Android NDK

#### Install NDK via Android Studio

1. Open Android Studio
2. Go to SDK Manager (Tools → SDK Manager)
3. Select "SDK Tools" tab
4. Check "NDK (Side by side)"
5. Click "Apply" to install

#### Set Environment Variable

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/26.1.10909125
```

Adjust the version number to match your installation. Check your NDK directory:

```bash
ls $HOME/Android/Sdk/ndk/
```

Reload your shell:

```bash
source ~/.zshrc  # or ~/.bashrc
```

### 2. Configure Cargo

Copy the template to the repository root:

```bash
cd /path/to/jolt-ios
cp jolt-android/cargo-config.toml.template .cargo/config.toml
```

Or add to existing `.cargo/config.toml`:

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

### 3. Build Native Libraries

From the repository root:

```bash
# Build with Java wrapper generation
JOLT_ANDROID_JAVA_OUT="$(pwd)/target/android/java" ./scripts/build-android.sh
```

This creates:
- `target/android/jniLibs/` - Ready to copy into Android project
- `target/android/java/JoltProver.java` - Java wrapper class

### 4. Create Android Project

If you don't have an Android project yet:

```bash
# Using Android Studio: File → New → New Project → Empty Activity
# Or create manually with Gradle
```

### 5. Integrate into Android Project

#### Copy Native Libraries

```bash
cd /path/to/your-android-app
cp -r /path/to/jolt-ios/target/android/jniLibs app/src/main/
```

Directory structure should be:

```
app/src/main/
├── jniLibs/
│   ├── arm64-v8a/
│   │   └── libjolt_android.so
│   ├── armeabi-v7a/
│   │   └── libjolt_android.so
│   ├── x86_64/
│   │   └── libjolt_android.so
│   └── x86/
│       └── libjolt_android.so
├── java/
│   └── com/yourapp/...
└── AndroidManifest.xml
```

#### Copy Java Wrapper

```bash
mkdir -p app/src/main/java/com/jolt
cp /path/to/jolt-ios/target/android/java/JoltProver.java app/src/main/java/com/jolt/
```

### 6. Use in Your App

#### Kotlin Example

```kotlin
package com.yourapp

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.jolt.JoltProver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        testJoltIntegration()
    }
    
    private fun testJoltIntegration() {
        // Test basic functionality
        val version = JoltProver.getVersion()
        val buildInfo = JoltProver.getBuildInfo()
        
        Log.d("Jolt", "Version: $version")
        Log.d("Jolt", "Build: $buildInfo")
        
        // Test JNI is working
        val result = JoltProver.testAdd(5, 7)
        Log.d("Jolt", "5 + 7 = $result")  // Should be 12
        
        // Try proof generation (currently returns not implemented)
        tryProofGeneration()
    }
    
    private fun tryProofGeneration() {
        lifecycleScope.launch(Dispatchers.IO) {
            // Load ELF binary from assets
            val elfBytes = assets.open("program.elf").use { it.readBytes() }
            val inputBytes = "Hello, Jolt!".toByteArray()
            
            val result = JoltProver.prove(
                elfBytes,
                inputBytes,
                1024 * 1024,  // 1 MB max input
                1024 * 1024   // 1 MB max output
            )
            
            withContext(Dispatchers.Main) {
                when {
                    result.isNotImplemented() -> {
                        Log.w("Jolt", "Proof generation not yet implemented")
                        // Expected for now
                    }
                    result.isSuccess() -> {
                        Log.d("Jolt", "Proof generated!")
                        Log.d("Jolt", "Cycles: ${result.cycles}")
                        Log.d("Jolt", "Proof size: ${result.proofData.size} bytes")
                        // Save or transmit proof
                    }
                    else -> {
                        Log.e("Jolt", "Error: ${result.status}")
                    }
                }
            }
        }
    }
}
```

#### Java Example

```java
package com.yourapp;

import android.os.Bundle;
import android.util.Log;
import androidx.appcompat.app.AppCompatActivity;
import com.jolt.JoltProver;
import java.io.InputStream;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "Jolt";
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        // Test on background thread
        Executors.newSingleThreadExecutor().execute(() -> {
            String version = JoltProver.getVersion();
            Log.d(TAG, "Version: " + version);
            
            int sum = JoltProver.testAdd(10, 20);
            Log.d(TAG, "10 + 20 = " + sum);
            
            try {
                byte[] elfBytes = loadElfFromAssets();
                byte[] input = "test".getBytes();
                
                JoltProver.ProofResult result = JoltProver.prove(
                    elfBytes, input, 1024*1024, 1024*1024
                );
                
                if (result.isNotImplemented()) {
                    Log.w(TAG, "Not yet implemented");
                } else if (result.isSuccess()) {
                    Log.d(TAG, "Cycles: " + result.cycles);
                }
            } catch (Exception e) {
                Log.e(TAG, "Error", e);
            }
        });
    }
    
    private byte[] loadElfFromAssets() throws Exception {
        try (InputStream is = getAssets().open("program.elf")) {
            return is.readAllBytes();  // API 33+, use alternative for older
        }
    }
}
```

## Build Configuration

### Gradle Configuration (Optional)

If you want Gradle to automatically pick up updated libraries:

```gradle
// app/build.gradle.kts

android {
    // ...
    
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("../../target/android/jniLibs")
        }
    }
}
```

### ABI Filters (Optional)

To reduce APK size, you can filter ABIs:

```gradle
android {
    defaultConfig {
        // Only include ARM64 for modern devices
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }
}
```

Or create separate APKs per ABI:

```gradle
android {
    splits {
        abi {
            isEnable = true
            reset()
            include("arm64-v8a", "armeabi-v7a")
            isUniversalApk = true
        }
    }
}
```

## Current Limitations

**The JNI functions are currently stubs.** They compile and link correctly, but:

- `prove()` returns `ERROR_NOT_IMPLEMENTED`
- `verify()` returns `ERROR_NOT_IMPLEMENTED`
- Only `getVersion()`, `getBuildInfo()`, and `testAdd()` work

See `jolt-android/README.md` for implementation roadmap.

## Testing

### Unit Tests

```bash
cargo test --package jolt-android
```

### Integration Test in Android

Create a test that verifies JNI is working:

```kotlin
@Test
fun testJoltJNI() {
    val version = JoltProver.getVersion()
    assertNotNull(version)
    assertTrue(version.isNotEmpty())
    
    val sum = JoltProver.testAdd(42, 58)
    assertEquals(100, sum)
}
```

### Device Testing

Run on various architectures:

- **ARM64** (most modern phones): Pixel, Galaxy S20+, etc.
- **ARMv7** (older phones): Still common in budget devices
- **x86_64** (emulator): Android Studio emulator on Intel/AMD Macs
- **x86** (older emulator): Legacy emulator images

## Performance Considerations

### Threading

Always run proof generation on background threads:

```kotlin
// Using Coroutines
lifecycleScope.launch(Dispatchers.Default) {
    val result = JoltProver.prove(...)
}

// Using WorkManager for long-running tasks
val workRequest = OneTimeWorkRequestBuilder<ProofWorker>()
    .setConstraints(
        Constraints.Builder()
            .setRequiresDeviceIdle(true)  // Only when device is idle
            .setRequiresCharging(true)    // Only when charging
            .build()
    )
    .build()
WorkManager.getInstance(context).enqueue(workRequest)
```

### Memory

Proof generation will use significant memory:

```kotlin
val runtime = Runtime.getRuntime()
val maxMemory = runtime.maxMemory()
val usedMemory = runtime.totalMemory() - runtime.freeMemory()

Log.d("Jolt", "Max memory: ${maxMemory / 1024 / 1024} MB")
Log.d("Jolt", "Used: ${usedMemory / 1024 / 1024} MB")
```

### Battery

For production apps, consider:

- Only generate proofs when plugged in
- Show progress notifications
- Allow users to cancel long operations

```kotlin
val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
if (!powerManager.isPowerSaveMode) {
    // Safe to do heavy computation
}
```

## Troubleshooting

### "Library not found" Error

Check library is in correct location:

```bash
ls app/src/main/jniLibs/arm64-v8a/libjolt_android.so
```

### "No implementation found for native method"

JNI function signatures must match exactly. The package name `com.jolt` must match the Rust function names:

```rust
Java_com_jolt_JoltProver_prove
// ^^^ ^^^^^^^^^ ^^^^^^^^^ ^^^^^
// |   |         |         method name
// |   |         class name
// |   package name
// Java prefix
```

If you change the package, update the Rust code accordingly.

### Build Fails with Linker Errors

1. Check `ANDROID_NDK_HOME` is set:
   ```bash
   echo $ANDROID_NDK_HOME
   ```

2. Verify NDK tools exist:
   ```bash
   ls $ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin/
   ```

3. Update `.cargo/config.toml` with correct API level

### APK Size Too Large

Each architecture adds 5-10 MB. To reduce size:

1. Use ABI splits (separate APKs per architecture)
2. Only include ARM64 for modern devices
3. Use Android App Bundle (AAB) format - Google Play handles ABI filtering

## Next Steps

1. Verify basic integration works (`testAdd()` returns correct value)
2. Implement proof generation in `jolt-android/src/lib.rs` (see README.md)
3. Test with real RISC-V ELF binaries
4. Profile performance on target devices
5. Optimize for battery and memory usage
6. Add progress notifications for long operations
7. Handle edge cases (low memory, process death, etc.)

## Example Project Structure

```
your-android-app/
├── app/
│   ├── build.gradle.kts
│   └── src/
│       ├── main/
│       │   ├── AndroidManifest.xml
│       │   ├── java/
│       │   │   └── com/
│       │   │       ├── jolt/
│       │   │       │   └── JoltProver.java  ← Copy here
│       │   │       └── yourapp/
│       │   │           └── MainActivity.kt
│       │   ├── jniLibs/  ← Copy here
│       │   │   ├── arm64-v8a/
│       │   │   │   └── libjolt_android.so
│       │   │   ├── armeabi-v7a/
│       │   │   │   └── libjolt_android.so
│       │   │   └── ...
│       │   └── res/
│       └── test/
└── build.gradle.kts
```

## Resources

- [Android NDK Documentation](https://developer.android.com/ndk)
- [JNI Tips](https://developer.android.com/training/articles/perf-jni)
- [rust-android-gradle Plugin](https://github.com/mozilla/rust-android-gradle) - Alternative integration approach
- [Rust on Android Guide](https://mozilla.github.io/firefox-browser-architecture/experiments/2017-09-21-rust-on-android.html)

## License

Follows the jolt-core licensing (see repository root).

