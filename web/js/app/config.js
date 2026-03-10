export const VERSION = '2.0.38';

// Model - relative path (served from same origin)
export const MODEL_BASE_URL = './models';

export const MODEL = {
  name: 'vesselboost.onnx',
  label: 'VesselBoost',
  numClasses: 1,
  patchSize: [64, 64, 64]
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

export const PIPELINE_STEPS = ['load', 'n4', 'bet', 'denoise', 'inference'];

if (typeof self !== 'undefined') self.VesselBoostConfig = { VERSION, MODEL_BASE_URL, MODEL, INFERENCE_DEFAULTS, VIEWER_CONFIG, PROGRESS_CONFIG, STAGE_NAMES, ONNX_CONFIG, CACHE_CONFIG, PIPELINE_STEPS };
