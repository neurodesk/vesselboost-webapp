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
- **Preprocessing**: N4ITK bias field correction + non-local means denoising (Rust/WASM)
- **Configurable**: target spacing, overlap, probability threshold, component size filtering
- **Privacy**: all processing happens locally in the browser

## Model Conversion

Convert VesselBoost PyTorch weights to ONNX:

```bash
# Download pre-trained weights
bash scripts/download_weights.sh

# Convert to ONNX
pip install torch onnx onnxruntime
python scripts/convert_model.py --checkpoint .tmp_weights/vesselboost_weights.pth

# Optional: quantize for smaller file size
python scripts/convert_model.py --checkpoint .tmp_weights/vesselboost_weights.pth --quantize
```

## Rust Preprocessing (Optional)

The N4ITK bias field correction and NLM denoising run as Rust compiled to WASM:

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
├── rust-preprocessing/    # Rust WASM crate (N4ITK + NLM)
├── scripts/               # Model conversion and version scripts
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
3. Resample to target spacing (default 0.3mm isotropic)
4. N4ITK bias field correction (WASM, optional)
5. Non-local means denoising (WASM, optional)
6. Z-score normalize
7. Crop foreground
8. 3D sliding window inference (ONNX Runtime Web)
9. Threshold probabilities
10. Remove small connected components
11. Inverse transforms -> output NIfTI

## Citations

If you use VesselBoost, please cite:

- **VesselBoost**: Marshall K, et al. VesselBoost: A Python Toolbox for Small Blood Vessel Segmentation. [GitHub](https://github.com/KMarshallX/VesselBoost/)
- **dcm2niix**: Li X, Morgan PS, Ashburner J, Smith J, Rorden C. The first step for neuroimaging data analysis: DICOM to NIfTI conversion. J Neurosci Methods. 2016;264:47-56. [GitHub](https://github.com/rordenlab/dcm2niix)
- **ONNX Runtime Web**: Microsoft. [onnxruntime.ai](https://onnxruntime.ai)
- **NiiVue**: NiiVue Contributors. [github.com/niivue/niivue](https://github.com/niivue/niivue)

## Privacy

All processing happens locally in your browser. No data is uploaded to any server.
