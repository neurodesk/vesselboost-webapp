#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../web/preprocessing-wasm"

echo "Building preprocessing WASM..."
cd "$SCRIPT_DIR"
wasm-pack build --target no-modules --out-dir "$OUTPUT_DIR" --release

# Clean up unnecessary files
rm -f "$OUTPUT_DIR/.gitignore" "$OUTPUT_DIR/package.json" "$OUTPUT_DIR/README.md"

echo "Done. WASM files in: $OUTPUT_DIR"
