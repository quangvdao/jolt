#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE="${JOLT_ANDROID_PACKAGE:-jolt-android}"
FEATURES="${JOLT_ANDROID_FEATURES:-}"
IFS=' ' read -r -a TARGETS <<< "${JOLT_ANDROID_TARGETS:-aarch64-linux-android armv7-linux-androideabi x86_64-linux-android i686-linux-android}"
BUILD_TYPE="${JOLT_ANDROID_BUILD_TYPE:-release}"
CARGO_FLAGS=("$@")

PROFILE_FLAG=()
ARTIFACT_DIR="$BUILD_TYPE"
case "$BUILD_TYPE" in
  release)
    PROFILE_FLAG=(--release)
    ARTIFACT_DIR=release
    ;;
  debug)
    PROFILE_FLAG=()
    ARTIFACT_DIR=debug
    ;;
  *)
    PROFILE_FLAG=(--profile "$BUILD_TYPE")
    ARTIFACT_DIR="$BUILD_TYPE"
    ;;
esac

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "No targets specified via JOLT_ANDROID_TARGETS" >&2
  exit 1
fi

# Check for Android NDK
if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
  echo "error: ANDROID_NDK_HOME not set" >&2
  echo "Please install Android NDK and set ANDROID_NDK_HOME environment variable" >&2
  echo "Example: export ANDROID_NDK_HOME=\$HOME/Android/Sdk/ndk/26.1.10909125" >&2
  exit 1
fi

if [[ ! -d "$ANDROID_NDK_HOME" ]]; then
  echo "error: ANDROID_NDK_HOME directory does not exist: $ANDROID_NDK_HOME" >&2
  exit 1
fi

echo "Using Android NDK: $ANDROID_NDK_HOME"

pushd "$REPO_ROOT" >/dev/null

for target in "${TARGETS[@]}"; do
  echo "==> Building $PACKAGE for $target ($BUILD_TYPE)"
  COMMAND=(cargo build --package "$PACKAGE" --lib --target "$target" "${PROFILE_FLAG[@]}")
  
  if [[ -n "$FEATURES" ]]; then
    COMMAND+=(--no-default-features --features "$FEATURES")
  fi
  
  if [[ ${#CARGO_FLAGS[@]} -gt 0 ]]; then
    COMMAND+=("${CARGO_FLAGS[@]}")
  fi
  
  "${COMMAND[@]}"
done

ANDROID_OUT_DIR="$REPO_ROOT/target/android"
mkdir -p "$ANDROID_OUT_DIR"

LIB_NAME="lib${PACKAGE//-/_}.so"

# Map Rust targets to Android ABIs
declare -A TARGET_TO_ABI
TARGET_TO_ABI[aarch64-linux-android]="arm64-v8a"
TARGET_TO_ABI[armv7-linux-androideabi]="armeabi-v7a"
TARGET_TO_ABI[x86_64-linux-android]="x86_64"
TARGET_TO_ABI[i686-linux-android]="x86"

echo ""
echo "==> Copying libraries to Android jniLibs structure"

for target in "${TARGETS[@]}"; do
  ABI="${TARGET_TO_ABI[$target]}"
  LIB_PATH="$REPO_ROOT/target/$target/$ARTIFACT_DIR/$LIB_NAME"
  
  if [[ -f "$LIB_PATH" ]]; then
    # Create both flat and jniLibs directory structures
    # Flat structure for reference
    cp "$LIB_PATH" "$ANDROID_OUT_DIR/${LIB_NAME%.so}-${target}.so"
    
    # jniLibs structure for direct Android integration
    JNILIBS_DIR="$ANDROID_OUT_DIR/jniLibs/$ABI"
    mkdir -p "$JNILIBS_DIR"
    cp "$LIB_PATH" "$JNILIBS_DIR/$LIB_NAME"
    
    echo "  ✓ $ABI: $LIB_NAME"
  else
    echo "  ✗ warning: expected artifact $LIB_PATH missing" >&2
  fi
done

# Generate Java wrapper if requested
if [[ -n "${JOLT_ANDROID_JAVA_OUT:-}" ]]; then
  echo ""
  echo "==> Generating Java wrapper"
  
  JAVA_OUT="$JOLT_ANDROID_JAVA_OUT"
  mkdir -p "$JAVA_OUT"
  
  cat > "$JAVA_OUT/JoltProver.java" << 'EOF'
package com.jolt;

/**
 * JNI wrapper for Jolt zero-knowledge proof system.
 * 
 * This class provides a Java interface to the Jolt prover and verifier
 * implemented in Rust.
 * 
 * Note: The prove() and verify() methods currently return "not implemented"
 * errors. See jolt-android/README.md for implementation details.
 */
public class JoltProver {
    static {
        System.loadLibrary("jolt_android");
    }

    /**
     * Get the version of the Jolt library.
     */
    public static native String getVersion();

    /**
     * Get build information about the Jolt library.
     */
    public static native String getBuildInfo();

    /**
     * Generate a zero-knowledge proof for a RISC-V program.
     * 
     * @param elfBytes The ELF binary of the RISC-V program
     * @param inputBytes Input data for the program
     * @param maxInputSize Maximum size for input buffer
     * @param maxOutputSize Maximum size for output buffer
     * @return ProofResult containing the proof data or error information
     */
    public static native ProofResult prove(
        byte[] elfBytes,
        byte[] inputBytes,
        int maxInputSize,
        int maxOutputSize
    );

    /**
     * Verify a zero-knowledge proof.
     * 
     * @param proofData Serialized proof data
     * @param preprocessingData Serialized preprocessing data
     * @param expectedOutput Expected output from the program
     * @return VerifyResult indicating whether the proof is valid
     */
    public static native VerifyResult verify(
        byte[] proofData,
        byte[] preprocessingData,
        byte[] expectedOutput
    );

    /**
     * Test function to verify JNI is working correctly.
     */
    public static native int testAdd(int a, int b);

    /**
     * Result of a proof generation operation.
     */
    public static class ProofResult {
        /** Status code (0 = success, 99 = not implemented, etc.) */
        public final int status;
        
        /** Number of cycles executed */
        public final long cycles;
        
        /** Serialized proof data (null if error) */
        public final byte[] proofData;

        public ProofResult(int status, long cycles, byte[] proofData) {
            this.status = status;
            this.cycles = cycles;
            this.proofData = proofData;
        }

        public boolean isSuccess() {
            return status == 0;
        }

        public boolean isNotImplemented() {
            return status == 99;
        }
    }

    /**
     * Result of a proof verification operation.
     */
    public static class VerifyResult {
        /** Status code (0 = success, 99 = not implemented, etc.) */
        public final int status;
        
        /** Whether the proof is valid */
        public final boolean isValid;

        public VerifyResult(int status, boolean isValid) {
            this.status = status;
            this.isValid = isValid;
        }

        public boolean isSuccess() {
            return status == 0;
        }

        public boolean isNotImplemented() {
            return status == 99;
        }
    }

    /**
     * Status codes returned by Jolt operations.
     */
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
EOF

  echo "  ✓ Generated: $JAVA_OUT/JoltProver.java"
fi

echo ""
echo "Build complete!"
echo ""
echo "Output directories:"
echo "  Libraries:  $ANDROID_OUT_DIR/jniLibs/"
echo ""
echo "To integrate into an Android project:"
echo "  1. Copy $ANDROID_OUT_DIR/jniLibs/ to your app/src/main/jniLibs/"
if [[ -n "${JOLT_ANDROID_JAVA_OUT:-}" ]]; then
  echo "  2. Copy $JAVA_OUT/JoltProver.java to your app/src/main/java/com/jolt/"
fi
echo "  3. Use JoltProver class from your Kotlin/Java code"
echo ""

popd >/dev/null

