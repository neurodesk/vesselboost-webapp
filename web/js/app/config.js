export const VERSION = '2.0.81';

// Model - relative path (served from same origin)
export const MODEL_BASE_URL = './models';

export const MODEL = {
  name: 'vesselboost.onnx',
  label: 'VesselBoost',
  numClasses: 1,
  patchSize: [64, 64, 64]
};

// Available VesselBoost model variants
// Each corresponds to a different training strategy from the VesselBoost Docker container
export const MODELS = [
  {
    id: 'manual',
    name: 'vesselboost.onnx',
    label: 'Default (TOF MRA)',
    description: 'Standard model trained on manual annotations. Recommended for most TOF MRA data.',
    numClasses: 1,
    patchSize: [64, 64, 64],
    dockerName: 'manual_0429'
  },
  {
    id: 'omelette1',
    name: 'vesselboost-omelette1.onnx',
    label: 'Boosted (Sensitive)',
    description: 'TTA-boosted model with higher sensitivity. May over-segment in noisy regions.',
    numClasses: 1,
    patchSize: [64, 64, 64],
    dockerName: 'omelette1_0429'
  },
  {
    id: 'omelette2',
    name: 'vesselboost-omelette2.onnx',
    label: 'Boosted (Moderate)',
    description: 'TTA-boosted model with moderate sensitivity. Balance between default and sensitive.',
    numClasses: 1,
    patchSize: [64, 64, 64],
    dockerName: 'omelette2_0429'
  },
  {
    id: 't2s',
    name: 'vesselboost-t2s.onnx',
    label: 'T2* / SWI',
    description: 'Trained on T2*-weighted data. Use for SWI or T2* acquisitions.',
    numClasses: 1,
    patchSize: [64, 64, 64],
    dockerName: 't2s_mod_ep1k2_0728'
  }
];

export const SYNTHSTRIP_MODEL = {
  name: 'synthstrip.onnx',
  label: 'SynthStrip',
  targetSpacing: [1.0, 1.0, 1.0]
};

export const SYNTHSTRIP_FAST_MODEL = {
  name: 'synthstrip.onnx',
  label: 'SynthStrip Fast',
  targetSpacing: [2.0, 2.0, 2.0]
};

export const INFERENCE_DEFAULTS = {
  cropForegroundMargin: 20,
  overlap: 0,
  probabilityThreshold: 0.1,
  minComponentSize: 10,
  biasCorrection: true,
  denoising: false,
  fractionalIntensity: 0.5
};

export const VIEWER_CONFIG = {
  loadingText: "",
  dragToMeasure: false,
  isColorbar: false,
  textHeight: 0.03,
  show3Dcrosshair: false,
  crosshairColor: [0.23, 0.51, 0.96, 1.0],
  crosshairWidth: 0.75
};

export const PROGRESS_CONFIG = {
  animationSpeed: 0.5
};

export const STAGE_NAMES = {
  'input': 'Input',
  'bet': 'Brain Extraction',
  'n4': 'Bias Correction',
  'nlm': 'Denoising',
  'segmentation': 'Segmentation'
};

export const ONNX_CONFIG = {
  executionProviders: ['webgpu', 'wasm'],
  graphOptimizationLevel: 'all'
};

export const CACHE_CONFIG = {
  name: 'VesselBoostModelCache',
  storeName: 'models',
  maxSizeMB: 500
};

export const PIPELINE_STEPS = ['load', 'n4', 'denoise', 'inference', 'bet'];

if (typeof self !== 'undefined') self.VesselBoostConfig = { VERSION, MODEL_BASE_URL, MODEL, MODELS, SYNTHSTRIP_MODEL, SYNTHSTRIP_FAST_MODEL, INFERENCE_DEFAULTS, VIEWER_CONFIG, PROGRESS_CONFIG, STAGE_NAMES, ONNX_CONFIG, CACHE_CONFIG, PIPELINE_STEPS };
