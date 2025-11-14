#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE="jolt-core"
FEATURES="${JOLT_IOS_FEATURES:-minimal prover}"
IFS=' ' read -r -a TARGETS <<< "${JOLT_IOS_TARGETS:-aarch64-apple-ios x86_64-apple-ios}"
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
  COMMAND=(cargo build
    --package "$PACKAGE"
    --lib
    --target "$target"
    "${PROFILE_FLAG[@]}"
    --no-default-features
    --features "$FEATURES")
  if [[ ${#CARGO_FLAGS[@]} -gt 0 ]]; then
    COMMAND+=("${CARGO_FLAGS[@]}")
  fi
  "${COMMAND[@]}"
done

IOS_OUT_DIR="$REPO_ROOT/target/ios"
mkdir -p "$IOS_OUT_DIR"

for target in "${TARGETS[@]}"; do
  LIB_PATH="$REPO_ROOT/target/$target/$ARTIFACT_DIR/libjolt_core.a"
  if [[ -f "$LIB_PATH" ]]; then
    cp "$LIB_PATH" "$IOS_OUT_DIR/libjolt_core-${target}.a"
  else
    echo "warning: expected artifact $LIB_PATH missing" >&2
  fi
done

if command -v xcodebuild >/dev/null && [[ -n "${JOLT_IOS_HEADERS:-}" ]]; then
  echo "==> Creating JoltCore.xcframework"
  XC_ARGS=(
    -library "$IOS_OUT_DIR/libjolt_core-aarch64-apple-ios.a" -headers "$JOLT_IOS_HEADERS"
  )
  if [[ -f "$IOS_OUT_DIR/libjolt_core-x86_64-apple-ios.a" ]]; then
    XC_ARGS+=(
      -library "$IOS_OUT_DIR/libjolt_core-x86_64-apple-ios.a" -headers "$JOLT_IOS_HEADERS"
    )
  fi
  xcodebuild -create-xcframework "${XC_ARGS[@]}" -output "$IOS_OUT_DIR/JoltCore.xcframework"
else
  echo "Skipping xcframework creation (set JOLT_IOS_HEADERS and ensure xcodebuild is available)" >&2
fi

popd >/dev/null
