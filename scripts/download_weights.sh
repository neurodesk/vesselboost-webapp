#!/bin/bash
# Extract pre-trained VesselBoost weights from the Docker container
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$SCRIPT_DIR/../.tmp_weights"

mkdir -p "$WEIGHTS_DIR"

echo "Extracting VesselBoost pre-trained weights from Docker container..."
echo "Target: $WEIGHTS_DIR/"

# The correct model weights are shipped in the VesselBoost Docker container.
# Available models in vnmd/vesselboost_2.0.0:
#   manual_0429    - default TOF MRA model (recommended)
#   omelette1_0429 - TTA-boosted model (more sensitive, may over-segment)
#   omelette2_0429 - TTA-boosted model (moderate sensitivity)
#   t2s_mod_ep1k2_0728 - T2*-weighted model (for SWI/T2* data)
#
# Note: The OSF download (BM_VB2_aug_all_ep2k_bat_10_0903) is a different
# model that produces highly fragmented results. Use the Docker models instead.

DOCKER_IMAGE="vnmd/vesselboost_2.0.0"

# Model name → output filename pairs (space-separated)
MODEL_PAIRS="
manual_0429:vesselboost_weights.pth
omelette1_0429:vesselboost_omelette1_weights.pth
omelette2_0429:vesselboost_omelette2_weights.pth
t2s_mod_ep1k2_0728:vesselboost_t2s_weights.pth
"

# Allow selecting a specific model via argument, or extract all
SELECTED_MODEL="${1:-all}"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is required to extract model weights."
    echo "Install Docker or manually copy the model from the VesselBoost container."
    exit 1
fi

# Pull the image if not present
if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null 2>&1; then
    echo "Pulling Docker image: $DOCKER_IMAGE"
    docker pull "$DOCKER_IMAGE"
fi

extract_model() {
    local docker_name="$1"
    local output_name="$2"

    echo ""
    echo "Extracting: $docker_name -> $output_name"
    docker run --rm -v "$WEIGHTS_DIR:/weights" "$DOCKER_IMAGE" \
        cp "/opt/VesselBoost/saved_models/$docker_name" "/weights/$output_name"
    echo "  Done: $WEIGHTS_DIR/$output_name"
}

found=0
for pair in $MODEL_PAIRS; do
    docker_name="${pair%%:*}"
    output_name="${pair##*:}"

    if [ "$SELECTED_MODEL" = "all" ] || [ "$SELECTED_MODEL" = "$docker_name" ]; then
        extract_model "$docker_name" "$output_name"
        found=1
    fi
done

if [ "$found" = "0" ]; then
    echo "Error: Unknown model '$SELECTED_MODEL'"
    echo "Available models: manual_0429 omelette1_0429 omelette2_0429 t2s_mod_ep1k2_0728"
    exit 1
fi

echo ""
echo "Extracted weights to: $WEIGHTS_DIR/"
echo ""
echo "To convert to ONNX, run:"
echo "  python scripts/convert_model.py --all"
echo ""
echo "Or convert individually:"
echo "  python scripts/convert_model.py --checkpoint $WEIGHTS_DIR/vesselboost_weights.pth"
echo "  python scripts/convert_model.py --checkpoint $WEIGHTS_DIR/vesselboost_omelette1_weights.pth --output web/models/vesselboost-omelette1.onnx"
echo "  python scripts/convert_model.py --checkpoint $WEIGHTS_DIR/vesselboost_omelette2_weights.pth --output web/models/vesselboost-omelette2.onnx"
echo "  python scripts/convert_model.py --checkpoint $WEIGHTS_DIR/vesselboost_t2s_weights.pth --output web/models/vesselboost-t2s.onnx"
