# VesselBoost Web App

Browser-based blood vessel segmentation using the [VesselBoost](https://github.com/KMarshallX/VesselBoost/) 3D UNet model. All processing runs entirely client-side using ONNX Runtime Web.

## Quick Start

```bash
# 1. Download ONNX Runtime WASM files
cd web
bash setup.sh

# 2. Place your ONNX model
# (See "Model Conversion" below, or place vesselboost.onnx in web/models/)

# 3. Start development server
bash run.sh
# Open http://localhost:8080
```

## Features

- **3D UNet inference** with sliding window (64x64x64 patches)
- **Binary vessel segmentation** (vessel/background)
- **DICOM and NIfTI** input support
- **Interactive pipeline**: each preprocessing step (N4, BET, denoising) can be run or skipped independently
- **Preprocessing**: N4ITK bias field correction, BET brain extraction, non-local means denoising (Rust/WASM)
- **Configurable**: overlap, probability threshold, component size filtering
- **Privacy**: all processing happens locally in the browser

## Model Weights

The default model (`manual_0429`) is extracted from the VesselBoost 2.0.0 Docker container. This is the recommended model for TOF MRA vessel segmentation.

```bash
# Extract weights from VesselBoost Docker container and convert to ONNX
bash scripts/download_weights.sh
pip install torch onnx onnxruntime
python scripts/convert_model.py --checkpoint .tmp_weights/vesselboost_weights.pth

# Optional: quantize for smaller file size
python scripts/convert_model.py --checkpoint .tmp_weights/vesselboost_weights.pth --quantize
```

### Available Models

The VesselBoost Docker container (`vnmd/vesselboost_2.0.0`) ships four models:

| Model | Description | Use case |
|-------|-------------|----------|
| `manual_0429` | Default TOF MRA model | Recommended for general use |
| `omelette1_0429` | TTA-boosted, high sensitivity | More vessels detected, may over-segment |
| `omelette2_0429` | TTA-boosted, moderate sensitivity | Balance between sensitivity and specificity |
| `t2s_mod_ep1k2_0728` | T2*-weighted model | For SWI/T2* data (not TOF MRA) |

To use a different model, extract it from the container and convert:

```bash
docker run --rm -v $(pwd)/.tmp_weights:/weights vnmd/vesselboost_2.0.0 \
    cp /opt/VesselBoost/saved_models/<model_name> /weights/vesselboost_weights.pth
python scripts/convert_model.py --checkpoint .tmp_weights/vesselboost_weights.pth
```

## Rust Preprocessing (Optional)

The N4ITK bias field correction, BET brain extraction, and NLM denoising run as Rust compiled to WASM:

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build
cd rust-preprocessing
bash build.sh
```

If not built, the app will skip preprocessing and still run inference.

## Project Structure

```
vesselboost-webapp/
├── .github/workflows/     # CI/CD (release + GitHub Pages deploy)
├── rust-preprocessing/    # Rust WASM crate (N4ITK + NLM + BET)
├── scripts/               # Model conversion, validation, and version scripts
├── web/
│   ├── js/
│   │   ├── app/           # Config and labels
│   │   ├── controllers/   # FileIO, DICOM, Inference, Viewer
│   │   ├── modules/       # UI components and inference pipeline
│   │   ├── vesselboost-app.js    # Main app
│   │   └── inference-worker.js   # Web Worker (3D inference pipeline)
│   ├── models/            # ONNX model files
│   ├── preprocessing-wasm/# Built WASM preprocessing output
│   └── index.html
└── README.md
```

## Pipeline

1. Parse NIfTI / convert DICOM
2. Orient to RAS
3. N4ITK bias field correction (WASM, optional — run or skip)
4. BET brain extraction (WASM, optional — run or skip)
5. Non-local means denoising (WASM, optional — run or skip)
6. Pad to 64-voxel multiples (nearest-neighbor zoom matching `scipy.ndimage.zoom`)
7. Z-score normalize
8. 3D sliding window inference (ONNX Runtime Web)
9. Threshold probabilities (default 0.1)
10. Inverse transforms (resize back to original dimensions)
11. Apply brain mask (if BET was used)
12. Remove small connected components
13. Inverse orient -> output NIfTI

## Validation

Compare the web app output against the Python VesselBoost pipeline:

```bash
pip install nibabel onnxruntime scipy numpy
python scripts/validate_onnx.py <nifti_path> web/models/vesselboost.onnx
```

## Citations

If you use VesselBoost, please cite:

- **VesselBoost**: Xu M, Ribeiro FL, Barth M, et al. VesselBoost: A Python Toolbox for Small Blood Vessel Segmentation in Human Magnetic Resonance Angiography Data. Aperture Neuro. 2024;4. doi:10.52294/001c.123217. [GitHub](https://github.com/KMarshallX/VesselBoost/)
- **dcm2niix**: Li X, Morgan PS, Ashburner J, Smith J, Rorden C. The first step for neuroimaging data analysis: DICOM to NIfTI conversion. J Neurosci Methods. 2016;264:47-56. [GitHub](https://github.com/rordenlab/dcm2niix)
- **ONNX Runtime Web**: Microsoft. [onnxruntime.ai](https://onnxruntime.ai)
- **NiiVue**: NiiVue Contributors. [github.com/niivue/niivue](https://github.com/niivue/niivue)

## Privacy

All processing happens locally in your browser. No data is uploaded to any server.
