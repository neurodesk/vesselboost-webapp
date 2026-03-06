#!/bin/bash
# One-time setup: download ONNX Runtime Web WASM files
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/wasm"

# ONNX Runtime Web
ORT_VERSION="1.21.0"
ORT_BASE="https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist"

echo "Downloading ONNX Runtime Web v${ORT_VERSION}..."

ORT_FILES=(
  ort.webgpu.min.js
  ort-wasm-simd-threaded.mjs
  ort-wasm-simd-threaded.wasm
  ort-wasm-simd-threaded.jsep.mjs
  ort-wasm-simd-threaded.jsep.wasm
)

for f in "${ORT_FILES[@]}"; do
  echo "  $f"
  curl -sL -o "$SCRIPT_DIR/wasm/$f" "$ORT_BASE/$f"
done

echo "Done. Files saved to wasm/"

# Build preprocessing WASM if rust-preprocessing/ exists and wasm-pack is installed
RUST_DIR="$SCRIPT_DIR/../rust-preprocessing"
if [[ -d "$RUST_DIR" ]] && command -v wasm-pack &>/dev/null; then
  echo ""
  echo "Building preprocessing WASM..."
  cd "$RUST_DIR"
  bash build.sh
  echo "Preprocessing WASM built and copied to web/preprocessing-wasm/"
else
  echo ""
  echo "Note: Preprocessing WASM not built (rust-preprocessing/ not found or wasm-pack not installed)"
  echo "  Install wasm-pack: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
fi

echo ""
echo "Note: Place your ONNX model file in: $SCRIPT_DIR/models/vesselboost.onnx"
