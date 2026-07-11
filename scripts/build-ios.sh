#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE="${JOLT_IOS_PACKAGE:-jolt-ios}"
FEATURES="${JOLT_IOS_FEATURES:-}"
IFS=' ' read -r -a TARGETS <<< "${JOLT_IOS_TARGETS:-aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios}"
BUILD_TYPE="${JOLT_IOS_BUILD_TYPE:-release}"
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
  echo "No targets specified via JOLT_IOS_TARGETS" >&2
  exit 1
fi

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

IOS_OUT_DIR="$REPO_ROOT/target/ios"
mkdir -p "$IOS_OUT_DIR"

LIB_NAME="lib${PACKAGE//-/_}.a"

for target in "${TARGETS[@]}"; do
  LIB_PATH="$REPO_ROOT/target/$target/$ARTIFACT_DIR/$LIB_NAME"
  if [[ -f "$LIB_PATH" ]]; then
    cp "$LIB_PATH" "$IOS_OUT_DIR/${LIB_NAME%.a}-${target}.a"
  else
    echo "warning: expected artifact $LIB_PATH missing" >&2
  fi
done

if command -v xcodebuild >/dev/null && [[ -n "${JOLT_IOS_HEADERS:-}" ]]; then
  echo "==> Creating JoltCore.xcframework"
  
  # Combine simulator libraries into a universal binary if both exist
  SIM_LIBS=()
  [[ -f "$IOS_OUT_DIR/${LIB_NAME%.a}-aarch64-apple-ios-sim.a" ]] && SIM_LIBS+=("$IOS_OUT_DIR/${LIB_NAME%.a}-aarch64-apple-ios-sim.a")
  [[ -f "$IOS_OUT_DIR/${LIB_NAME%.a}-x86_64-apple-ios.a" ]] && SIM_LIBS+=("$IOS_OUT_DIR/${LIB_NAME%.a}-x86_64-apple-ios.a")
  
  if [[ ${#SIM_LIBS[@]} -gt 1 ]]; then
    echo "==> Creating universal simulator library"
    lipo -create "${SIM_LIBS[@]}" -output "$IOS_OUT_DIR/${LIB_NAME%.a}-simulator.a"
    SIM_LIB="$IOS_OUT_DIR/${LIB_NAME%.a}-simulator.a"
  elif [[ ${#SIM_LIBS[@]} -eq 1 ]]; then
    SIM_LIB="${SIM_LIBS[0]}"
  else
    SIM_LIB=""
  fi
  
  # Build xcframework arguments
  XC_ARGS=()
  
  # Add device library
  if [[ -f "$IOS_OUT_DIR/${LIB_NAME%.a}-aarch64-apple-ios.a" ]]; then
    XC_ARGS+=(-library "$IOS_OUT_DIR/${LIB_NAME%.a}-aarch64-apple-ios.a" -headers "$JOLT_IOS_HEADERS")
  fi
  
  # Add simulator library
  if [[ -n "$SIM_LIB" ]]; then
    XC_ARGS+=(-library "$SIM_LIB" -headers "$JOLT_IOS_HEADERS")
  fi

  if [[ ${#XC_ARGS[@]} -gt 0 ]]; then
    rm -rf "$IOS_OUT_DIR/JoltCore.xcframework"
    xcodebuild -create-xcframework "${XC_ARGS[@]}" -output "$IOS_OUT_DIR/JoltCore.xcframework"
  else
    echo "warning: no libraries found to create xcframework" >&2
  fi
else
  echo "Skipping xcframework creation (set JOLT_IOS_HEADERS and ensure xcodebuild is available)" >&2
fi

popd >/dev/null
