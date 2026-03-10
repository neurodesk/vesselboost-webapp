/**
 * InferenceExecutor
 *
 * Handles Web Worker lifecycle for ONNX model inference.
 * Supports both step-by-step interactive pipeline and legacy single-run mode.
 */

import { VERSION } from '../app/config.js';

export class InferenceExecutor {
  constructor(options) {
    this.updateOutput = options.updateOutput || (() => {});
    this.setProgress = options.setProgress || (() => {});
    this.onStageData = options.onStageData || (() => {});
    this.onComplete = options.onComplete || (() => {});
    this.onError = options.onError || (() => {});
    this.onInitialized = options.onInitialized || (() => {});
    this.onStepComplete = options.onStepComplete || (() => {});
    this.onVolumeInfo = options.onVolumeInfo || (() => {});

    this.worker = null;
    this.workerReady = false;
    this.workerInitializing = false;
    this.running = false;
    this.webgpuAvailable = false;
    this.wasmAvailable = false;
    this.results = {};
    this.stageOrder = [];

    // Step status tracking
    this.stepStatus = {
      load: 'pending',
      n4: 'pending',
      bet: 'pending',
      denoise: 'pending',
      inference: 'pending'
    };
    this.volumeInfo = null;
  }

  isReady() { return this.workerReady; }
  isRunning() { return this.running; }

  hasResult(stage) { return !!this.results[stage]?.file; }
  getResult(stage) { return this.results[stage] || null; }
  getResults() { return this.results; }
  getStageOrder() { return this.stageOrder; }

  getStepStatus(step) { return this.stepStatus[step]; }
  getVolumeInfo() { return this.volumeInfo; }

  _setupWorker() {
    if (this.worker) return;

    this.worker = new Worker(`js/inference-worker.js?v=${VERSION}`);

    this.worker.onmessage = (e) => {
      const { type, ...data } = e.data;

      switch (type) {
        case 'progress':
          this.setProgress(data.value, data.text);
          break;
        case 'log':
          this.updateOutput(data.message);
          break;
        case 'error':
          this._handleError(data.message);
          break;
        case 'initialized':
          this.workerReady = true;
          this.workerInitializing = false;
          this.webgpuAvailable = !!data.webgpuAvailable;
          this.wasmAvailable = !!data.wasmPreprocessingAvailable;
          this.updateOutput('ONNX Runtime ready');
          this.onInitialized();
          break;
        case 'complete':
          this._handleComplete();
          break;
        case 'stageData':
          this._handleStageData(data);
          break;
        case 'step-complete':
          this._handleStepComplete(data.step);
          break;
        case 'volume-info':
          this._handleVolumeInfo(data);
          break;
      }
    };

    this.worker.onerror = (e) => {
      this.updateOutput(`Worker error: ${e.message}`);
      console.error('Worker error:', e);
      this._handleError(e.message);
    };
  }

  _handleError(message) {
    this.updateOutput(`Error: ${message}`);
    this.setProgress(0, 'Failed');
    this.running = false;
    this.onError(message);
  }

  _handleComplete() {
    this.updateOutput('Segmentation completed successfully!');
    this.running = false;
    this.onComplete();
  }

  _handleStageData(data) {
    if (!this.stageOrder.includes(data.stage)) {
      this.stageOrder.push(data.stage);
    }

    const blob = new Blob([data.niftiData], { type: 'application/octet-stream' });
    const file = new File([blob], `${data.stage}.nii`, { type: 'application/octet-stream' });
    this.results[data.stage] = {
      file: file,
      description: data.description
    };

    Promise.resolve(this.onStageData(data)).catch(err => {
      console.error('Error handling stage data:', err);
      this.updateOutput(`Error displaying ${data.stage}: ${err.message}`);
    });
  }

  _handleStepComplete(step) {
    // Preserve 'skipped' status if already set by skip method
    if (this.stepStatus[step] !== 'skipped') {
      this.stepStatus[step] = 'complete';
    }
    this.running = false;
    this.onStepComplete(step);
  }

  _handleVolumeInfo(data) {
    this.volumeInfo = {
      rasDims: data.rasDims,
      rasSpacing: data.rasSpacing,
      totalSlices: data.totalSlices
    };
    this.onVolumeInfo(this.volumeInfo);
  }

  async initialize() {
    this._setupWorker();

    if (this.workerReady) return;

    if (this.workerInitializing) {
      return new Promise((resolve) => {
        const checkReady = setInterval(() => {
          if (this.workerReady) {
            clearInterval(checkReady);
            resolve();
          }
        }, 100);
      });
    }

    this.workerInitializing = true;
    this.updateOutput('Initializing ONNX Runtime...');

    this.worker.postMessage({ type: 'init', version: VERSION });

    return new Promise((resolve) => {
      const checkReady = setInterval(() => {
        if (this.workerReady) {
          clearInterval(checkReady);
          resolve();
        }
      }, 100);
    });
  }

  // ==================== Step Methods ====================

  async loadVolume(inputData) {
    await this.initialize();
    this.running = true;
    this.stepStatus.load = 'running';
    this.worker.postMessage(
      { type: 'load', data: { inputData } },
      [inputData]
    );
  }

  async runN4() {
    await this.initialize();
    this.running = true;
    this.stepStatus.n4 = 'running';
    this.worker.postMessage({ type: 'run-n4' });
  }

  skipN4() {
    this.stepStatus.n4 = 'skipped';
    this.running = true;
    this.worker.postMessage({ type: 'skip-n4' });
  }

  async runBET(fractionalIntensity, method = 'bet', modelBaseUrl) {
    await this.initialize();
    this.running = true;
    this.stepStatus.bet = 'running';
    this.worker.postMessage({ type: 'run-bet', data: { fractionalIntensity, method, modelBaseUrl } });
  }

  skipBET() {
    this.stepStatus.bet = 'skipped';
    this.running = true;
    this.worker.postMessage({ type: 'skip-bet' });
  }

  async runDenoise() {
    await this.initialize();
    this.running = true;
    this.stepStatus.denoise = 'running';
    this.worker.postMessage({ type: 'run-denoise' });
  }

  skipDenoise() {
    this.stepStatus.denoise = 'skipped';
    this.running = true;
    this.worker.postMessage({ type: 'skip-denoise' });
  }

  async runInference(settings) {
    await this.initialize();
    this.running = true;
    this.stepStatus.inference = 'running';
    this.worker.postMessage({ type: 'run-inference', data: settings });
  }

  async resetWorkerState() {
    await this.initialize();
    this.worker.postMessage({ type: 'reset-state' });
    this.stepStatus = {
      load: 'pending',
      n4: 'pending',
      bet: 'pending',
      denoise: 'pending',
      inference: 'pending'
    };
    this.volumeInfo = null;
    this.results = {};
    this.stageOrder = [];
  }

  // Reset downstream steps when a step is re-run
  resetDownstream(fromStep) {
    const steps = ['load', 'n4', 'bet', 'denoise', 'inference'];
    const idx = steps.indexOf(fromStep);
    if (idx < 0) return;
    for (let i = idx + 1; i < steps.length; i++) {
      // BET re-run does NOT invalidate downstream (mask is independent)
      if (fromStep === 'bet') break;
      this.stepStatus[steps[i]] = 'pending';
    }
  }

  // ==================== Legacy Methods ====================

  async run(config) {
    try {
      await this.initialize();

      this.updateOutput('Starting segmentation...');
      this.running = true;
      this.results = {};
      this.stageOrder = [];

      this.worker.postMessage({
        type: 'run',
        data: config
      }, config.inputData ? [config.inputData] : []);

      return true;
    } catch (error) {
      this._handleError(error.message);
      return false;
    }
  }

  cancel() {
    if (!this.running) return;

    this.updateOutput('Cancelling...');

    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
      this.workerReady = false;
      this.workerInitializing = false;
    }

    this.running = false;
    this.setProgress(0, 'Cancelled');
    this.updateOutput('Cancelled. Worker will be reinitialized on next action.');
  }

  removeResult(stage) {
    delete this.results[stage];
    this.stageOrder = this.stageOrder.filter(s => s !== stage);
  }

  clearResults() {
    this.results = {};
    this.stageOrder = [];
  }

  async downloadStage(stage) {
    if (!this.results[stage]?.file) {
      this.updateOutput(`${stage} not available`);
      return;
    }

    const file = this.results[stage].file;
    const url = URL.createObjectURL(file);
    const a = document.createElement('a');
    a.href = url;
    a.download = file.name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  downloadAll() {
    for (const stage of this.stageOrder) {
      this.downloadStage(stage);
    }
  }
}
