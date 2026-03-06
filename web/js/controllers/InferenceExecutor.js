/**
 * InferenceExecutor
 *
 * Handles Web Worker lifecycle for ONNX model inference.
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

    this.worker = null;
    this.workerReady = false;
    this.workerInitializing = false;
    this.running = false;
    this.webgpuAvailable = false;
    this.results = {};
    this.stageOrder = [];
  }

  isReady() { return this.workerReady; }
  isRunning() { return this.running; }

  hasResult(stage) { return !!this.results[stage]?.file; }
  getResult(stage) { return this.results[stage] || null; }
  getResults() { return this.results; }
  getStageOrder() { return this.stageOrder; }

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
          this.updateOutput('ONNX Runtime ready');
          this.onInitialized();
          break;
        case 'complete':
          this._handleComplete();
          break;
        case 'stageData':
          this._handleStageData(data);
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
    if (!this.running) return;

    if (!this.stageOrder.includes(data.stage)) {
      this.stageOrder.push(data.stage);
    }

    const blob = new Blob([data.niftiData], { type: 'application/octet-stream' });
    const file = new File([blob], `${data.stage}.nii`, { type: 'application/octet-stream' });
    this.results[data.stage] = {
      file: file,
      description: data.description
    };

    this.onStageData(data);
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

    this.updateOutput('Cancelling segmentation...');

    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
      this.workerReady = false;
      this.workerInitializing = false;
    }

    this.running = false;
    this.setProgress(0, 'Cancelled');
    this.updateOutput('Segmentation cancelled. Worker will be reinitialized on next run.');
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
