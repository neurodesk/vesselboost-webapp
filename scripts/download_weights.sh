#!/bin/bash
# Download pre-trained VesselBoost weights from OSF
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$SCRIPT_DIR/../.tmp_weights"

mkdir -p "$WEIGHTS_DIR"

echo "Downloading VesselBoost pre-trained weights from OSF..."
echo "Target: $WEIGHTS_DIR/"

# Download from OSF
OSF_URL="https://osf.io/abk4p/download"
curl -L -o "$WEIGHTS_DIR/vesselboost_weights.pth" "$OSF_URL"

echo ""
echo "Downloaded weights to: $WEIGHTS_DIR/vesselboost_weights.pth"
echo ""
echo "To convert to ONNX, run:"
echo "  python scripts/convert_model.py --checkpoint $WEIGHTS_DIR/vesselboost_weights.pth"
