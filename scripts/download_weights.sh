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
MODEL_NAME="manual_0429"

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

# Extract the model
docker run --rm -v "$WEIGHTS_DIR:/weights" "$DOCKER_IMAGE" \
    cp "/opt/VesselBoost/saved_models/$MODEL_NAME" "/weights/vesselboost_weights.pth"

echo ""
echo "Extracted weights to: $WEIGHTS_DIR/vesselboost_weights.pth"
echo "Model: $MODEL_NAME (from $DOCKER_IMAGE)"
echo ""
echo "To convert to ONNX, run:"
echo "  python scripts/convert_model.py --checkpoint $WEIGHTS_DIR/vesselboost_weights.pth"
