/**
 * VesselBoost - Browser-based blood vessel segmentation
 *
 * Main application class. Orchestrates controllers, viewer, and inference.
 * Pipeline is split into interactive steps that the user runs sequentially.
 */

import { FileIOController } from './controllers/FileIOController.js';
import { ViewerController } from './controllers/ViewerController.js';
import { InferenceExecutor } from './controllers/InferenceExecutor.js';
import { ConsoleOutput } from './modules/ui/ConsoleOutput.js';
import { ProgressManager } from './modules/ui/ProgressManager.js';
import { ModalManager } from './modules/ui/ModalManager.js';
import * as Config from './app/config.js';
import { generateNiivueColormap, getLabelName } from './app/labels.js';
import { computeAutoWindow } from './modules/ui/percentile.js';

class VesselBoostApp {
  constructor() {
    // NiiVue
    this.nv = new niivue.Niivue({
      ...Config.VIEWER_CONFIG,
      onLocationChange: (data) => {
        this._lastLocationData = data;
        this.updateViewerInfo(data);
      }
    });

    // UI modules
    this.console = new ConsoleOutput('consoleOutput');
    this.progress = new ProgressManager(Config.PROGRESS_CONFIG);

    // State
    this.inputFile = null;
    this.currentResultTab = 'input';
    this.currentRunningStep = null;
    this.abortUICheckpoint = null;
    this._inputVisible = true;
    this._overlaySliderValue = 0.5;
    this._segmentationVisible = true;
    this._lastLocationData = null;

    this.init();
  }

  async init() {
    // Version display
    const versionEl = document.getElementById('appVersion');
    if (versionEl) versionEl.textContent = `v${Config.VERSION}`;
    const footerVersionEl = document.getElementById('footerVersion');
    if (footerVersionEl) footerVersionEl.textContent = `v${Config.VERSION}`;
    const aboutVersionEl = document.getElementById('aboutAppVersion');
    if (aboutVersionEl) aboutVersionEl.textContent = `v${Config.VERSION}`;

    // Controllers
    this.fileIOController = new FileIOController({
      updateOutput: (msg) => this.updateOutput(msg),
      onFileLoaded: (file) => this.onFileLoaded(file)
    });

    this.viewerController = new ViewerController({
      nv: this.nv,
      updateOutput: (msg) => this.updateOutput(msg)
    });

    this.inferenceExecutor = new InferenceExecutor({
      updateOutput: (msg) => this.updateOutput(msg),
      setProgress: (val, text) => this.setProgress(val, text),
      onStageData: (data) => this.handleStageData(data),
      onComplete: () => this.onInferenceComplete(),
      onError: (msg) => this.onInferenceError(msg),
      onInitialized: () => this.onWorkerInitialized(),
      onStepComplete: (step) => this.onStepComplete(step),
      onVolumeInfo: (info) => this.onVolumeInfo(info),
      onBrainMaskOverlay: (file) => { this._brainMaskOverlayFile = file; }
    });

    // Modals
    this.aboutModal = new ModalManager('aboutModal');
    this.citationsModal = new ModalManager('citationsModal');
    this.privacyModal = new ModalManager('privacyModal');

    // Register custom colormap
    const colormapData = generateNiivueColormap();

    // Setup
    await this.setupViewer();

    // Register colormap after viewer is ready
    this.viewerController.registerVesselColormap(colormapData);

    this.setupEventListeners();
    this.setupInfoTooltips();

    // Start ONNX initialization in background
    this.inferenceExecutor.initialize();
  }

  async setupViewer() {
    await this.nv.attachTo('gl1');
    this.nv.setMultiplanarPadPixels(5);
    this.nv.setSliceType(this.nv.sliceTypeMultiplanar);
    this.nv.setInterpolation(true);
    this.nv.drawScene();
  }

  // ==================== Viewer Footer ====================

  updateViewerInfo(data) {
    const primaryEl = document.getElementById('viewerInfoPrimary');
    if (primaryEl) {
      primaryEl.textContent = data?.string || '';
    }

    const labelEl = document.getElementById('viewerInfoLabel');
    if (labelEl) {
      labelEl.textContent = this.getOverlayLabelText(data);
    }
  }

  getOverlayLabelText(data) {
    if (!this._segmentationVisible) return '';
    if (!this.nv?.volumes || this.nv.volumes.length < 2) return '';

    const rawValue = data?.values?.[1]?.value;
    if (!Number.isFinite(rawValue)) return '';

    const labelIndex = Math.round(rawValue);
    if (labelIndex <= 0) return '';

    return getLabelName(labelIndex);
  }

  // ==================== Event Listeners ====================

  setupEventListeners() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
      fileInput.addEventListener('change', (e) => {
        this.fileIOController.handleFiles(e.target.files);
      });
    }

    this.setupDropZone();

    // Step buttons
    const runDownsample = document.getElementById('runDownsampleBtn');
    if (runDownsample) runDownsample.addEventListener('click', () => this.runDownsample());

    const skipDownsample = document.getElementById('skipDownsampleBtn');
    if (skipDownsample) skipDownsample.addEventListener('click', () => this.skipDownsample());

    const runN4 = document.getElementById('runN4Btn');
    if (runN4) runN4.addEventListener('click', () => this.runN4());

    const skipN4 = document.getElementById('skipN4Btn');
    if (skipN4) skipN4.addEventListener('click', () => this.skipN4());

    const runBET = document.getElementById('runBETBtn');
    if (runBET) runBET.addEventListener('click', () => this.runBET());

    const skipBET = document.getElementById('skipBETBtn');
    if (skipBET) skipBET.addEventListener('click', () => this.skipBET());

    const applyBET = document.getElementById('applyBETBtn');
    if (applyBET) applyBET.addEventListener('click', () => this.applyBrainMask());

    const dilateBET = document.getElementById('dilateBETBtn');
    if (dilateBET) dilateBET.addEventListener('click', () => this.dilateBrainMask());

    const erodeBET = document.getElementById('erodeBETBtn');
    if (erodeBET) erodeBET.addEventListener('click', () => this.erodeBrainMask());

    // BET method dropdown: toggle FI visibility
    const betMethodSelect = document.getElementById('betMethodSelect');
    if (betMethodSelect) {
      betMethodSelect.addEventListener('change', () => {
        const fiGroup = document.getElementById('betFiGroup');
        if (fiGroup) {
          fiGroup.style.display = betMethodSelect.value.startsWith('synthstrip') ? 'none' : '';
        }
      });
    }

    const runDenoise = document.getElementById('runDenoiseBtn');
    if (runDenoise) runDenoise.addEventListener('click', () => this.runDenoise());

    const skipDenoise = document.getElementById('skipDenoiseBtn');
    if (skipDenoise) skipDenoise.addEventListener('click', () => this.skipDenoise());

    const runBtn = document.getElementById('runSegmentation');
    if (runBtn) runBtn.addEventListener('click', () => this.runSegmentation());

    ['abortN4Btn', 'abortDenoiseBtn', 'abortInferenceBtn', 'abortBETBtn'].forEach(id => {
      const btn = document.getElementById(id);
      if (btn) btn.addEventListener('click', () => this.abortCurrentStep());
    });

    const cancelBtn = document.getElementById('cancelButton');
    if (cancelBtn) cancelBtn.addEventListener('click', () => this.abortCurrentStep());

    const copyConsole = document.getElementById('copyConsole');
    if (copyConsole) copyConsole.addEventListener('click', () => this.console.copyToClipboard());

    const clearConsole = document.getElementById('clearConsole');
    if (clearConsole) clearConsole.addEventListener('click', () => this.console.clear());

    document.querySelectorAll('.view-tab[data-view]').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.view-tab[data-view]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.viewerController.setViewType(btn.dataset.view);
      });
    });

    const opacitySlider = document.getElementById('overlayOpacity');
    if (opacitySlider) {
      opacitySlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        this._overlaySliderValue = val;
        if (this._segmentationVisible) {
          this.viewerController.setOverlayOpacity(val);
        }
        const display = document.getElementById('overlayOpacityValue');
        if (display) display.textContent = `${Math.round(val * 100)}%`;
      });
    }

    this.setupWindowControls();

    const interpToggle = document.getElementById('interpolation');
    if (interpToggle) {
      interpToggle.addEventListener('change', (e) => {
        this.nv.setInterpolation(!e.target.checked);
        this.nv.drawScene();
      });
    }

    const colorbarToggle = document.getElementById('colorbarToggle');
    if (colorbarToggle) {
      colorbarToggle.addEventListener('change', (e) => {
        this.nv.opts.isColorbar = e.target.checked;
        this.nv.drawScene();
      });
    }

    const crosshairToggle = document.getElementById('crosshairToggle');
    if (crosshairToggle) {
      crosshairToggle.addEventListener('change', (e) => {
        this.nv.setCrosshairWidth(e.target.checked ? 1 : 0);
      });
    }

    const downloadBtn = document.getElementById('downloadCurrentVolume');
    if (downloadBtn) {
      downloadBtn.addEventListener('click', () => this.downloadCurrentVolume());
    }

    const screenshotBtn = document.getElementById('screenshotViewer');
    if (screenshotBtn) {
      screenshotBtn.addEventListener('click', () => this.saveScreenshot());
    }

    const colormapSelect = document.getElementById('colormapSelect');
    if (colormapSelect) {
      colormapSelect.addEventListener('change', (e) => {
        if (this.nv.volumes?.length) {
          this.nv.volumes[0].colormap = e.target.value;
          this.nv.updateGLVolume();
        }
      });
    }

    const clearResults = document.getElementById('clearResults');
    if (clearResults) clearResults.addEventListener('click', () => this.clearResults());

    // Modal buttons
    const aboutBtn = document.getElementById('aboutButton');
    if (aboutBtn) aboutBtn.addEventListener('click', () => this.aboutModal.open());
    const closeAbout = document.getElementById('closeAbout');
    if (closeAbout) closeAbout.addEventListener('click', () => this.aboutModal.close());

    const citationsBtn = document.getElementById('citationsButton');
    if (citationsBtn) citationsBtn.addEventListener('click', () => this.citationsModal.open());
    const closeCitations = document.getElementById('closeCitations');
    if (closeCitations) closeCitations.addEventListener('click', () => this.citationsModal.close());

    const privacyBtn = document.getElementById('privacyButton');
    if (privacyBtn) privacyBtn.addEventListener('click', () => this.privacyModal.open());
    const closePrivacy = document.getElementById('closePrivacy');
    if (closePrivacy) closePrivacy.addEventListener('click', () => this.privacyModal.close());
  }

  setupDropZone() {
    const zone = document.getElementById('inputDropZone');
    if (!zone) return;

    zone.addEventListener('dragover', (e) => {
      e.preventDefault();
      zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => {
      zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('dragover');
      this.fileIOController.handleDropItems(e.dataTransfer.items);
    });
  }

  setupInfoTooltips() {
    document.querySelectorAll('.info-icon').forEach(icon => {
      const tooltip = icon.querySelector('.info-tooltip');
      if (!tooltip) return;

      icon.addEventListener('mouseenter', () => {
        tooltip.style.display = 'block';
        const iconRect = icon.getBoundingClientRect();
        const tipRect = tooltip.getBoundingClientRect();
        let top = iconRect.top - tipRect.height - 6;
        let left = iconRect.left + iconRect.width / 2 - tipRect.width / 2;
        if (top < 4) top = iconRect.bottom + 6;
        left = Math.max(4, Math.min(left, window.innerWidth - tipRect.width - 4));
        tooltip.style.top = `${top}px`;
        tooltip.style.left = `${left}px`;
      });

      icon.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
      });
    });
  }

  // ==================== Viewer Controls ====================

  setupWindowControls() {
    const rangeMin = document.getElementById('rangeMin');
    const rangeMax = document.getElementById('rangeMax');
    const windowMin = document.getElementById('windowMin');
    const windowMax = document.getElementById('windowMax');
    const resetBtn = document.getElementById('resetWindow');
    if (!rangeMin || !rangeMax || !windowMin || !windowMax) return;

    const updateSelected = () => {
      const selected = document.getElementById('rangeSelected');
      if (!selected) return;
      const min = parseFloat(rangeMin.value);
      const max = parseFloat(rangeMax.value);
      selected.style.left = `${min}%`;
      selected.style.width = `${max - min}%`;
    };

    const applyFromSliders = () => {
      if (!this.nv.volumes.length) return;
      const vol = this.nv.volumes[0];
      const dataMin = vol.global_min ?? 0;
      const dataMax = vol.global_max ?? 1;
      const range = dataMax - dataMin || 1;
      const newMin = dataMin + (parseFloat(rangeMin.value) / 100) * range;
      const newMax = dataMin + (parseFloat(rangeMax.value) / 100) * range;
      windowMin.value = newMin.toPrecision(4);
      windowMax.value = newMax.toPrecision(4);
      vol.cal_min = newMin;
      vol.cal_max = newMax;
      this.nv.updateGLVolume();
      updateSelected();
    };

    const applyFromInputs = () => {
      if (!this.nv.volumes.length) return;
      const vol = this.nv.volumes[0];
      const newMin = parseFloat(windowMin.value);
      const newMax = parseFloat(windowMax.value);
      if (isNaN(newMin) || isNaN(newMax)) return;
      vol.cal_min = newMin;
      vol.cal_max = newMax;
      this.nv.updateGLVolume();
      this.syncSlidersToVolume();
    };

    rangeMin.addEventListener('input', () => {
      if (parseFloat(rangeMin.value) > parseFloat(rangeMax.value) - 1) {
        rangeMin.value = parseFloat(rangeMax.value) - 1;
      }
      applyFromSliders();
    });

    rangeMax.addEventListener('input', () => {
      if (parseFloat(rangeMax.value) < parseFloat(rangeMin.value) + 1) {
        rangeMax.value = parseFloat(rangeMin.value) + 1;
      }
      applyFromSliders();
    });

    windowMin.addEventListener('change', applyFromInputs);
    windowMax.addEventListener('change', applyFromInputs);

    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        if (!this.nv.volumes.length) return;
        this.applyAutoContrast();
      });
    }
  }

  applyAutoContrast() {
    if (!this.nv.volumes.length) return;
    const vol = this.nv.volumes[0];

    // computeAutoWindow operates on vol.img which may be raw typed data
    // (e.g. Int16Array), but vol.cal_min/cal_max are in scaled space
    // (after NIfTI scl_slope/scl_inter). Convert using global_min/max.
    const { low, high, min: rawMin, max: rawMax } = computeAutoWindow(vol.img);

    let scaledLow = low;
    let scaledHigh = high;
    const rawRange = rawMax - rawMin;
    const scaledRange = vol.global_max - vol.global_min;
    if (rawRange > 0 && scaledRange > 0) {
      // Linear mapping: raw → scaled
      const slope = scaledRange / rawRange;
      const inter = vol.global_min - rawMin * slope;
      scaledLow = low * slope + inter;
      scaledHigh = high * slope + inter;
    }

    vol.cal_min = scaledLow;
    vol.cal_max = scaledHigh;
    this.nv.updateGLVolume();
    this.syncWindowControls();
  }

  syncWindowControls() {
    if (!this.nv.volumes.length) return;
    const vol = this.nv.volumes[0];
    const windowMin = document.getElementById('windowMin');
    const windowMax = document.getElementById('windowMax');
    if (windowMin) windowMin.value = (vol.cal_min ?? 0).toPrecision(4);
    if (windowMax) windowMax.value = (vol.cal_max ?? 1).toPrecision(4);
    this.syncSlidersToVolume();
    const dlBtn = document.getElementById('downloadCurrentVolume');
    if (dlBtn) dlBtn.disabled = false;
  }

  syncSlidersToVolume() {
    if (!this.nv.volumes.length) return;
    const vol = this.nv.volumes[0];
    const dataMin = vol.global_min ?? 0;
    const dataMax = vol.global_max ?? 1;
    const range = dataMax - dataMin || 1;
    const rangeMin = document.getElementById('rangeMin');
    const rangeMax = document.getElementById('rangeMax');
    const selected = document.getElementById('rangeSelected');
    if (!rangeMin || !rangeMax) return;
    const pctMin = Math.max(0, Math.min(100, ((vol.cal_min - dataMin) / range) * 100));
    const pctMax = Math.max(0, Math.min(100, ((vol.cal_max - dataMin) / range) * 100));
    rangeMin.value = pctMin;
    rangeMax.value = pctMax;
    if (selected) {
      selected.style.left = `${pctMin}%`;
      selected.style.width = `${pctMax - pctMin}%`;
    }
  }

  downloadCurrentVolume() {
    if (!this.nv.volumes?.length) {
      this.updateOutput('No volume loaded');
      return;
    }
    const vol = this.nv.volumes[0];
    const name = (vol.name || 'volume').replace(/\.(nii|nii\.gz)$/i, '');
    const niftiBuffer = this.createNiftiFromVolume(vol);
    const blob = new Blob([niftiBuffer], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${name}.nii`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    this.updateOutput(`Downloaded: ${name}.nii`);
  }

  createNiftiFromVolume(vol) {
    const hdr = vol.hdr;
    const img = vol.img;
    let datatype = 16, bitpix = 32, bytesPerVoxel = 4;
    if (img instanceof Float64Array) { datatype = 64; bitpix = 64; bytesPerVoxel = 8; }
    else if (img instanceof Int16Array) { datatype = 4; bitpix = 16; bytesPerVoxel = 2; }
    else if (img instanceof Uint8Array) { datatype = 2; bitpix = 8; bytesPerVoxel = 1; }

    const headerSize = 352;
    const buffer = new ArrayBuffer(headerSize + img.length * bytesPerVoxel);
    const view = new DataView(buffer);

    view.setInt32(0, 348, true);
    const dims = hdr.dims || [3, vol.dims[1], vol.dims[2], vol.dims[3], 1, 1, 1, 1];
    for (let i = 0; i < 8; i++) view.setInt16(40 + i * 2, dims[i] || 0, true);
    view.setInt16(70, datatype, true);
    view.setInt16(72, bitpix, true);
    const pixdim = hdr.pixDims || [1, 1, 1, 1, 1, 1, 1, 1];
    for (let i = 0; i < 8; i++) view.setFloat32(76 + i * 4, pixdim[i] || 1, true);
    view.setFloat32(108, headerSize, true);
    view.setFloat32(112, hdr.scl_slope || 1, true);
    view.setFloat32(116, hdr.scl_inter || 0, true);
    view.setUint8(123, 10);
    view.setInt16(252, hdr.qform_code || 1, true);
    view.setInt16(254, hdr.sform_code || 1, true);
    if (hdr.affine) {
      for (let i = 0; i < 4; i++) {
        view.setFloat32(280 + i * 4, hdr.affine[0][i] || 0, true);
        view.setFloat32(296 + i * 4, hdr.affine[1][i] || 0, true);
        view.setFloat32(312 + i * 4, hdr.affine[2][i] || 0, true);
      }
    }
    view.setUint8(344, 0x6E);
    view.setUint8(345, 0x2B);
    view.setUint8(346, 0x31);
    view.setUint8(347, 0x00);

    new Uint8Array(buffer, headerSize).set(new Uint8Array(img.buffer, img.byteOffset, img.byteLength));
    return buffer;
  }

  saveScreenshot() {
    let filename = 'vesselboost_screenshot.png';
    if (this.nv.volumes?.length) {
      const name = (this.nv.volumes[0].name || 'volume').replace(/\.(nii|nii\.gz)$/i, '');
      filename = `${name}_screenshot.png`;
    }
    this.nv.saveScene(filename);
    this.updateOutput(`Screenshot saved: ${filename}`);
  }

  // ==================== File Handling ====================

  async onFileLoaded(file) {
    await this.resetForNewFile();
    this.inputFile = file;
    this._inputVisible = true;
    await this.viewerController.loadBaseVolume(file);
    this.applyDefaultBaseColormap();
    this.syncWindowControls();
    this.applyAutoContrast();

    // Send data to worker for loading
    const inputData = await file.arrayBuffer();
    this.setStepRunning('load');
    await this.inferenceExecutor.loadVolume(inputData);
  }

  async resetForNewFile() {
    if (this.inferenceExecutor.isRunning()) {
      this.inferenceExecutor.cancel();
    }

    this.inputFile = null;
    this.currentResultTab = 'input';
    this.currentRunningStep = null;
    this.abortUICheckpoint = null;
    this._inputVisible = true;
    this._segmentationVisible = true;
    this._overlaySliderValue = 0.5;
    this._lastLocationData = null;

    this.console.clear();
    this.progress.reset();
    this.resetStatusDisplay();
    this.resetProcessingInputs();
    this.resetViewerControls();

    await this.resetAllSteps();
    this.updateViewerInfo(null);
  }

  captureAbortUICheckpoint(step) {
    const sectionEnabled = {};
    const buttonsEnabled = {};

    for (const pipelineStep of ['n4', 'denoise', 'inference', 'bet']) {
      sectionEnabled[pipelineStep] = this.isStepEnabled(pipelineStep);
      buttonsEnabled[pipelineStep] = this.areStepButtonsEnabled(pipelineStep);
    }

    const applyGroup = document.getElementById('applyBETGroup');
    const applyBtn = document.getElementById('applyBETBtn');

    return {
      step,
      sectionEnabled,
      buttonsEnabled,
      applyBETVisible: applyGroup ? !applyGroup.classList.contains('hidden') : false,
      applyBETDisabled: applyBtn ? applyBtn.disabled : false,
      currentResultTab: this.currentResultTab || 'input',
      inputVisible: this._inputVisible,
      segmentationVisible: this._segmentationVisible,
      overlaySliderValue: this._overlaySliderValue
    };
  }

  beginAbortableStep(step) {
    this.currentRunningStep = step;
    this.abortUICheckpoint = this.captureAbortUICheckpoint(step);
    this.inferenceExecutor.captureCheckpoint(step);
  }

  async abortCurrentStep() {
    if (!this.currentRunningStep || !this.inferenceExecutor.isRunning()) return;

    const abortedStep = this.currentRunningStep;
    const checkpoint = this.abortUICheckpoint;
    const statusText = document.getElementById('statusText');
    if (statusText) statusText.textContent = 'Aborting...';

    this.resetAbortControls();

    try {
      const restoreResult = await this.inferenceExecutor.abortCurrentStep();
      if (!restoreResult) return;
      await this.restoreUIFromAbortCheckpoint(checkpoint, abortedStep);
    } catch (error) {
      this.onInferenceError(error?.message || String(error));
    } finally {
      this.currentRunningStep = null;
      this.abortUICheckpoint = null;
    }
  }

  async restoreUIFromAbortCheckpoint(checkpoint, abortedStep) {
    this.progress.reset();
    this.resetAbortControls();

    for (const step of Config.PIPELINE_STEPS) {
      this.updateStepBadge(step, this.inferenceExecutor.getStepStatus(step));
      if (checkpoint?.sectionEnabled?.[step] !== undefined) {
        this.setStepEnabled(step, checkpoint.sectionEnabled[step]);
      }
      if (checkpoint?.buttonsEnabled?.[step] !== undefined) {
        this.setStepButtonsEnabled(step, checkpoint.buttonsEnabled[step]);
      }
    }

    if (abortedStep) {
      this.updateStepBadge(abortedStep, 'pending');
      this.setStepButtonsEnabled(abortedStep, true);
    }

    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
      if (this.inferenceExecutor.getStageOrder().length > 0) {
        resultsSection.classList.remove('hidden');
        resultsSection.classList.remove('collapsed');
      } else {
        resultsSection.classList.add('hidden');
        resultsSection.classList.add('collapsed');
      }
    }

    const overlayControl = document.getElementById('overlayControl');
    if (overlayControl) {
      overlayControl.classList.toggle('hidden', !this.inferenceExecutor.getResult('segmentation'));
    }

    const applyGroup = document.getElementById('applyBETGroup');
    if (applyGroup) {
      applyGroup.classList.toggle('hidden', !checkpoint?.applyBETVisible);
    }
    const applyBtn = document.getElementById('applyBETBtn');
    if (applyBtn) {
      applyBtn.disabled = checkpoint?.applyBETDisabled ?? false;
    }

    this.currentResultTab = checkpoint?.currentResultTab || 'input';
    this._segmentationVisible = checkpoint?.segmentationVisible ?? true;
    this._overlaySliderValue = checkpoint?.overlaySliderValue ?? 0.5;

    this.rebuildResultsList();

    const targetStage = (this.currentResultTab === 'input' || this.inferenceExecutor.getResult(this.currentResultTab))
      ? this.currentResultTab
      : 'input';

    if (this.inputFile && targetStage) {
      await this.viewStage(targetStage);
    }

    this._inputVisible = checkpoint?.inputVisible ?? true;
    this.viewerController.setBaseOpacity(this._inputVisible ? 1 : 0);

    const opacitySlider = document.getElementById('overlayOpacity');
    if (opacitySlider) {
      opacitySlider.disabled = !this._segmentationVisible;
      opacitySlider.value = String(this._overlaySliderValue);
    }
    const opacityDisplay = document.getElementById('overlayOpacityValue');
    if (opacityDisplay) opacityDisplay.textContent = `${Math.round(this._overlaySliderValue * 100)}%`;

    if (this._segmentationVisible) {
      this.viewerController.setOverlayOpacity(this._overlaySliderValue);
    } else {
      this.viewerController.setOverlayOpacity(0);
    }

    const statusText = document.getElementById('statusText');
    if (statusText) statusText.textContent = 'Ready';
    this.updateViewerInfo(this._lastLocationData);
  }

  // ==================== Pipeline Step Methods ====================

  updateDownsampleInfo() {
    const info = this.inferenceExecutor.getVolumeInfo();
    const el = document.getElementById('downsampleInfo');
    if (!el || !info) return;
    const dims = info.rasDims;
    const spacing = info.rasSpacing;
    const totalVoxels = dims[0] * dims[1] * dims[2];
    el.textContent = `Resolution: ${dims[0]}x${dims[1]}x${dims[2]} (${spacing.map(v => v.toFixed(3)).join('x')}mm), ${(totalVoxels / 1e6).toFixed(1)}M voxels`;
  }

  async runDownsample() {
    if (this.inferenceExecutor.isRunning()) return;
    const factorSelect = document.getElementById('downsampleFactor');
    const factor = factorSelect ? parseInt(factorSelect.value) : 2;
    this.setStepRunning('downsample');
    await this.inferenceExecutor.downsample(factor);
  }

  async skipDownsample() {
    if (this.inferenceExecutor.isRunning()) return;
    // Remove downsample result and restore viewer to input
    this.inferenceExecutor.removeResult('downsample');
    this.inferenceExecutor.skipDownsample();
    if (this.inputFile) {
      await this.viewerController.loadBaseVolume(this.inputFile);
      this.currentResultTab = 'input';
      this._inputVisible = true;
      this.applyDefaultBaseColormap();
      this.syncWindowControls();
      this.applyAutoContrast();
    }
  }

  async runN4() {
    if (this.inferenceExecutor.isRunning()) return;
    this.beginAbortableStep('n4');
    this.setStepRunning('n4');
    this.inferenceExecutor.resetDownstream('n4');
    this.resetUIDownstream('n4');
    await this.inferenceExecutor.runN4();
  }

  async skipN4() {
    if (this.inferenceExecutor.isRunning()) return;
    // Remove N4 result and restore viewer to downsample result or input
    this.inferenceExecutor.removeResult('n4');
    this.inferenceExecutor.skipN4();
    const downsampleResult = this.inferenceExecutor.getResult('downsample');
    const baseFile = downsampleResult?.file || this.inputFile;
    if (baseFile) {
      await this.viewerController.loadBaseVolume(baseFile);
      this.currentResultTab = downsampleResult?.file ? 'downsample' : 'input';
      this._inputVisible = true;
      this.applyDefaultBaseColormap();
      this.syncWindowControls();
      this.applyAutoContrast();
    }
  }

  async runBET() {
    if (this.inferenceExecutor.isRunning()) return;
    const methodSelect = document.getElementById('betMethodSelect');
    const method = methodSelect ? methodSelect.value : 'bet';
    const betFiInput = document.getElementById('betFiInput');
    const fi = betFiInput ? parseFloat(betFiInput.value) : 0.5;
    this.beginAbortableStep('bet');
    this.setStepRunning('bet');
    const modelBaseUrl = new URL(Config.MODEL_BASE_URL, window.location.href).href;
    await this.inferenceExecutor.runBET(fi, method, modelBaseUrl);
  }

  async skipBET() {
    if (this.inferenceExecutor.isRunning()) return;
    // Remove BET result and let onStepComplete('bet') restore the segmentation view
    this.inferenceExecutor.removeResult('bet');
    this.inferenceExecutor.skipBET();
  }

  async applyBrainMask() {
    if (this.inferenceExecutor.isRunning()) return;
    const applyBtn = document.getElementById('applyBETBtn');
    if (applyBtn) applyBtn.disabled = true;
    await this.inferenceExecutor.applyBrainMask();
  }

  async dilateBrainMask() {
    if (this.inferenceExecutor.isRunning()) return;
    const dilateBtn = document.getElementById('dilateBETBtn');
    if (dilateBtn) dilateBtn.disabled = true;
    await this.inferenceExecutor.dilateBrainMask(1);
  }

  async erodeBrainMask() {
    if (this.inferenceExecutor.isRunning()) return;
    const erodeBtn = document.getElementById('erodeBETBtn');
    if (erodeBtn) erodeBtn.disabled = true;
    await this.inferenceExecutor.erodeBrainMask(1);
  }

  async runDenoise() {
    if (this.inferenceExecutor.isRunning()) return;
    const methodSelect = document.getElementById('denoiseMethodSelect');
    const method = methodSelect ? methodSelect.value : 'bilateral';
    this.beginAbortableStep('denoise');
    this.setStepRunning('denoise');
    this.inferenceExecutor.resetDownstream('denoise');
    this.resetUIDownstream('denoise');
    await this.inferenceExecutor.runDenoise(method);
  }

  async skipDenoise() {
    if (this.inferenceExecutor.isRunning()) return;
    // Remove denoise result and restore viewer
    this.inferenceExecutor.removeResult('nlm');
    this.inferenceExecutor.skipDenoise();
    // Reload the latest preprocessing base (N4 > downsample > input)
    const n4Result = this.inferenceExecutor.getResult('n4');
    const downsampleResult = this.inferenceExecutor.getResult('downsample');
    const baseFile = n4Result?.file || downsampleResult?.file || this.inputFile;
    if (baseFile) {
      await this.viewerController.loadBaseVolume(baseFile);
      this.currentResultTab = n4Result?.file ? 'n4' : (downsampleResult?.file ? 'downsample' : 'input');
      this._inputVisible = true;
      this.applyDefaultBaseColormap();
      this.syncWindowControls();
      this.applyAutoContrast();
    }
  }

  async runSegmentation() {
    if (this.inferenceExecutor.isRunning()) return;

    const overlapSelect = document.getElementById('overlapSelect');
    const overlap = overlapSelect ? parseFloat(overlapSelect.value) : Config.INFERENCE_DEFAULTS.overlap;

    const thresholdInput = document.getElementById('thresholdInput');
    const threshold = thresholdInput ? parseFloat(thresholdInput.value) : Config.INFERENCE_DEFAULTS.probabilityThreshold;

    const minSizeInput = document.getElementById('minSizeInput');
    const minComponentSize = minSizeInput ? parseInt(minSizeInput.value, 10) : Config.INFERENCE_DEFAULTS.minComponentSize;

    // Get selected model from dropdown
    const modelSelect = document.getElementById('modelSelect');
    const selectedModelId = modelSelect ? modelSelect.value : 'manual';
    const selectedModel = Config.MODELS.find(m => m.id === selectedModelId) || Config.MODELS[0];

    const modelBaseUrl = new URL(Config.MODEL_BASE_URL, window.location.href).href;
    this.beginAbortableStep('inference');

    // Clear previous results
    this.inferenceExecutor.clearResults();
    this.disableAllResultTabs();

    this.setStepRunning('inference');
    await this.inferenceExecutor.runInference({
      overlap,
      threshold,
      minComponentSize,
      modelName: selectedModel.name,
      patchSize: selectedModel.patchSize,
      modelBaseUrl
    });
  }

  // ==================== Step UI Management ====================

  setStepRunning(step) {
    this.updateStepBadge(step, 'running');
    this.setStepButtonsEnabled(step, false);
    this.setStepAbortVisible(step, true);
    if (this.currentRunningStep === step) {
      const cancelBtn = document.getElementById('cancelButton');
      if (cancelBtn) cancelBtn.disabled = false;
    }
  }

  getStepSectionId(step) {
    const sectionMap = {
      'load': null,
      'downsample': 'stepDownsampleSection',
      'n4': 'stepN4Section',
      'bet': 'stepBETSection',
      'denoise': 'stepDenoiseSection',
      'inference': 'stepInferenceSection'
    };
    return sectionMap[step] || null;
  }

  getStepButtonIds(step) {
    const buttonMap = {
      'downsample': ['runDownsampleBtn', 'skipDownsampleBtn'],
      'n4': ['runN4Btn', 'skipN4Btn'],
      'bet': ['runBETBtn', 'skipBETBtn'],
      'denoise': ['runDenoiseBtn', 'skipDenoiseBtn'],
      'inference': ['runSegmentation']
    };
    return buttonMap[step] || [];
  }

  getStepAbortButtonId(step) {
    const abortButtonMap = {
      'n4': 'abortN4Btn',
      'bet': 'abortBETBtn',
      'denoise': 'abortDenoiseBtn',
      'inference': 'abortInferenceBtn'
    };
    return abortButtonMap[step] || null;
  }

  isStepEnabled(step) {
    const sectionId = this.getStepSectionId(step);
    if (!sectionId) return false;
    const section = document.getElementById(sectionId);
    return !!section && !section.classList.contains('step-disabled');
  }

  areStepButtonsEnabled(step) {
    const buttonIds = this.getStepButtonIds(step);
    if (buttonIds.length === 0) return false;
    return buttonIds.every(id => {
      const btn = document.getElementById(id);
      return !!btn && !btn.disabled;
    });
  }

  setStepAbortVisible(step, visible) {
    const abortButtonId = this.getStepAbortButtonId(step);
    if (!abortButtonId) return;

    const abortBtn = document.getElementById(abortButtonId);
    if (!abortBtn) return;

    abortBtn.classList.toggle('hidden', !visible);
    abortBtn.disabled = !visible;
  }

  resetAbortControls() {
    for (const step of ['n4', 'denoise', 'inference', 'bet']) {
      this.setStepAbortVisible(step, false);
    }

    const cancelBtn = document.getElementById('cancelButton');
    if (cancelBtn) cancelBtn.disabled = true;
  }

  async resetAllSteps() {
    // Reset worker state
    if (this.inferenceExecutor.isReady()) {
      await this.inferenceExecutor.resetWorkerState();
    }

    this.currentRunningStep = null;
    this.abortUICheckpoint = null;

    // Reset all UI step sections
    for (const step of Config.PIPELINE_STEPS) {
      this.updateStepBadge(step, '');
      this.setStepEnabled(step, false);
      this.setStepButtonsEnabled(step, false);
      this.setStepAbortVisible(step, false);
    }

    // Reset results
    this.inferenceExecutor.clearResults();
    this.disableAllResultTabs();

    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
      resultsSection.classList.add('hidden');
      resultsSection.classList.add('collapsed');
    }

    const overlayControl = document.getElementById('overlayControl');
    if (overlayControl) overlayControl.classList.add('hidden');

    const applyGroup = document.getElementById('applyBETGroup');
    if (applyGroup) applyGroup.classList.add('hidden');

    this.resetAbortControls();
  }

  resetStatusDisplay() {
    const statusText = document.getElementById('statusText');
    if (statusText) statusText.textContent = 'Ready';
    this.resetAbortControls();
  }

  resetProcessingInputs() {
    // Reset downsample
    const downsampleFactor = document.getElementById('downsampleFactor');
    if (downsampleFactor) downsampleFactor.value = '2';
    const downsampleInfo = document.getElementById('downsampleInfo');
    if (downsampleInfo) downsampleInfo.textContent = 'Loading volume info...';

    // Reset BET method to SynthStrip (default)
    const betMethodSelect = document.getElementById('betMethodSelect');
    if (betMethodSelect) {
      betMethodSelect.value = 'synthstrip';
      const fiGroup = document.getElementById('betFiGroup');
      if (fiGroup) fiGroup.style.display = 'none';
    }

    const betFiInput = document.getElementById('betFiInput');
    if (betFiInput) betFiInput.value = String(Config.INFERENCE_DEFAULTS.fractionalIntensity);

    const overlapSelect = document.getElementById('overlapSelect');
    if (overlapSelect) overlapSelect.value = String(Config.INFERENCE_DEFAULTS.overlap);

    const thresholdInput = document.getElementById('thresholdInput');
    if (thresholdInput) thresholdInput.value = String(Config.INFERENCE_DEFAULTS.probabilityThreshold);

    const minSizeInput = document.getElementById('minSizeInput');
    if (minSizeInput) minSizeInput.value = String(Config.INFERENCE_DEFAULTS.minComponentSize);
  }

  resetViewerControls() {
    document.querySelectorAll('.view-tab[data-view]').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.view === 'multiplanar');
    });
    this.viewerController.setViewType('multiplanar');

    const rangeMin = document.getElementById('rangeMin');
    if (rangeMin) rangeMin.value = '0';
    const rangeMax = document.getElementById('rangeMax');
    if (rangeMax) rangeMax.value = '100';
    const rangeSelected = document.getElementById('rangeSelected');
    if (rangeSelected) {
      rangeSelected.style.left = '0%';
      rangeSelected.style.width = '100%';
    }

    const windowMin = document.getElementById('windowMin');
    if (windowMin) windowMin.value = '';
    const windowMax = document.getElementById('windowMax');
    if (windowMax) windowMax.value = '';

    const overlayControl = document.getElementById('overlayControl');
    if (overlayControl) overlayControl.classList.add('hidden');

    const applyGroup = document.getElementById('applyBETGroup');
    if (applyGroup) applyGroup.classList.add('hidden');

    const opacitySlider = document.getElementById('overlayOpacity');
    if (opacitySlider) {
      opacitySlider.disabled = false;
      opacitySlider.value = '0.5';
    }
    const opacityDisplay = document.getElementById('overlayOpacityValue');
    if (opacityDisplay) opacityDisplay.textContent = '50%';

    const colormapSelect = document.getElementById('colormapSelect');
    if (colormapSelect) colormapSelect.value = 'gray';

    const interpolationToggle = document.getElementById('interpolation');
    if (interpolationToggle) interpolationToggle.checked = false;
    this.nv.setInterpolation(true);

    const colorbarToggle = document.getElementById('colorbarToggle');
    if (colorbarToggle) colorbarToggle.checked = false;
    this.nv.opts.isColorbar = false;

    const crosshairToggle = document.getElementById('crosshairToggle');
    if (crosshairToggle) crosshairToggle.checked = true;
    this.nv.setCrosshairWidth(Config.VIEWER_CONFIG.crosshairWidth ?? 1);

    const downloadBtn = document.getElementById('downloadCurrentVolume');
    if (downloadBtn) downloadBtn.disabled = true;

    this.nv.drawScene();
  }

  applyDefaultBaseColormap() {
    const colormapSelect = document.getElementById('colormapSelect');
    const colormap = colormapSelect?.value || 'gray';
    if (!this.nv.volumes?.length) return;
    this.nv.volumes[0].colormap = colormap;
    this.nv.updateGLVolume();
  }

  async onStepComplete(step) {
    const status = this.inferenceExecutor.getStepStatus(step);
    this.updateStepBadge(step, status);
    this.setStepButtonsEnabled(step, true);
    this.setStepAbortVisible(step, false);
    this.resetAbortControls();

    if (this.currentRunningStep === step) {
      this.currentRunningStep = null;
      this.abortUICheckpoint = null;
    }

    const statusText = document.getElementById('statusText');
    if (statusText) statusText.textContent = 'Ready';

    // Enable next step section
    switch (step) {
      case 'load':
        this.setStepEnabled('downsample', true);
        this.setStepButtonsEnabled('downsample', true);
        this.updateDownsampleInfo();
        break;
      case 'downsample':
        this.setStepEnabled('n4', true);
        this.setStepButtonsEnabled('n4', true);
        break;
      case 'n4':
        this.setStepEnabled('denoise', true);
        this.setStepButtonsEnabled('denoise', true);
        break;
      case 'denoise':
        this.setStepEnabled('inference', true);
        this.setStepButtonsEnabled('inference', true);
        break;
      case 'inference':
        // Handled by onInferenceComplete — enable BET after segmentation
        this.setStepEnabled('bet', true);
        this.setStepButtonsEnabled('bet', true);
        break;
      case 'bet': {
        const applyGroup = document.getElementById('applyBETGroup');
        if (status === 'skipped') {
          // Hide apply button when BET is skipped/undone
          if (applyGroup) applyGroup.classList.add('hidden');
          // Show segmentation overlay (reverted to previous mask state)
          const segResult = this.inferenceExecutor.getResult('segmentation');
          const segFile = segResult?.file;
          const downsampleResult = this.inferenceExecutor.getResult('downsample');
          const betBaseFile = downsampleResult?.file || this.inputFile;
          if (segFile && betBaseFile) {
            await this.viewerController.showResultAsOverlay(betBaseFile, segFile, 'red');
            this._inputVisible = false;
            this.viewerController.setBaseOpacity(0);
            this._segmentationVisible = true;
            this._overlaySliderValue = 1.0;
            this.viewerController.setOverlayOpacity(1.0);
            const opacitySlider = document.getElementById('overlayOpacity');
            if (opacitySlider) opacitySlider.value = 1.0;
            const opacityDisplay = document.getElementById('overlayOpacityValue');
            if (opacityDisplay) opacityDisplay.textContent = '100%';
            this.syncWindowControls();
          }
        } else {
          // Show brain extraction result with segmentation overlay
          const betResult = this.inferenceExecutor.getResult('bet');
          const segResult = this.inferenceExecutor.getResult('segmentation');
          if (betResult?.file) {
            await this.viewerController.loadBaseVolume(betResult.file);
            this.currentResultTab = 'bet';
            this._inputVisible = true;
            this.applyDefaultBaseColormap();
            this.syncWindowControls();
            this.applyAutoContrast();
            // Overlay segmentation on the brain-extracted image
            if (segResult?.file) {
              await this.viewerController.loadOverlay(segResult.file, 'red', 0.5);
              this._segmentationVisible = true;
              this._overlaySliderValue = 0.5;
              const opacitySlider = document.getElementById('overlayOpacity');
              if (opacitySlider) opacitySlider.value = 0.5;
              const opacityDisplay = document.getElementById('overlayOpacityValue');
              if (opacityDisplay) opacityDisplay.textContent = '50%';
            }
          }
          if (applyGroup) applyGroup.classList.remove('hidden');
          const applyBtn = document.getElementById('applyBETBtn');
          if (applyBtn) applyBtn.disabled = false;
          const dilateBtn = document.getElementById('dilateBETBtn');
          if (dilateBtn) dilateBtn.disabled = false;
          const erodeBtn = document.getElementById('erodeBETBtn');
          if (erodeBtn) erodeBtn.disabled = false;
        }
        this.rebuildResultsList();
        break;
      }
      case 'apply-brain-mask': {
        // Show brain-extracted image as base with updated segmentation as overlay
        const betResult = this.inferenceExecutor.getResult('bet');
        const segResult = this.inferenceExecutor.getResult('segmentation');
        if (betResult?.file) {
          await this.viewerController.loadBaseVolume(betResult.file);
          this.currentResultTab = 'bet';
          this._inputVisible = true;
          this.applyDefaultBaseColormap();
          this.syncWindowControls();
          this.applyAutoContrast();
          if (segResult?.file) {
            await this.viewerController.loadOverlay(segResult.file, 'red', 0.5);
            this._segmentationVisible = true;
            this._overlaySliderValue = 0.5;
            const opacitySlider = document.getElementById('overlayOpacity');
            if (opacitySlider) opacitySlider.value = 0.5;
            const opacityDisplay = document.getElementById('overlayOpacityValue');
            if (opacityDisplay) opacityDisplay.textContent = '50%';
          }
        }
        this.rebuildResultsList();
        // Keep dilation/erosion controls visible for further adjustments
        const dilateBtn = document.getElementById('dilateBETBtn');
        if (dilateBtn) dilateBtn.disabled = false;
        const erodeBtn = document.getElementById('erodeBETBtn');
        if (erodeBtn) erodeBtn.disabled = false;
        const applyBtn = document.getElementById('applyBETBtn');
        if (applyBtn) applyBtn.disabled = false;
        break;
      }
      case 'dilate-brain-mask': {
        // Refresh the BET preview with dilated mask + segmentation overlay
        const betResult = this.inferenceExecutor.getResult('bet');
        const segResult = this.inferenceExecutor.getResult('segmentation');
        if (betResult?.file) {
          await this.viewerController.loadBaseVolume(betResult.file);
          this.currentResultTab = 'bet';
          this._inputVisible = true;
          this.applyDefaultBaseColormap();
          this.syncWindowControls();
          this.applyAutoContrast();
          if (segResult?.file) {
            await this.viewerController.loadOverlay(segResult.file, 'red', 0.5);
            this._segmentationVisible = true;
            this._overlaySliderValue = 0.5;
            const opacitySlider = document.getElementById('overlayOpacity');
            if (opacitySlider) opacitySlider.value = 0.5;
            const opacityDisplay = document.getElementById('overlayOpacityValue');
            if (opacityDisplay) opacityDisplay.textContent = '50%';
          }
        }
        const dilateBtn = document.getElementById('dilateBETBtn');
        if (dilateBtn) dilateBtn.disabled = false;
        const erodeBtn = document.getElementById('erodeBETBtn');
        if (erodeBtn) erodeBtn.disabled = false;
        const applyBtn = document.getElementById('applyBETBtn');
        if (applyBtn) applyBtn.disabled = false;
        break;
      }
      case 'erode-brain-mask': {
        // Refresh the BET preview with eroded mask + segmentation overlay
        const betResult = this.inferenceExecutor.getResult('bet');
        const segResult = this.inferenceExecutor.getResult('segmentation');
        if (betResult?.file) {
          await this.viewerController.loadBaseVolume(betResult.file);
          this.currentResultTab = 'bet';
          this._inputVisible = true;
          this.applyDefaultBaseColormap();
          this.syncWindowControls();
          this.applyAutoContrast();
          if (segResult?.file) {
            await this.viewerController.loadOverlay(segResult.file, 'red', 0.5);
            this._segmentationVisible = true;
            this._overlaySliderValue = 0.5;
            const opacitySlider = document.getElementById('overlayOpacity');
            if (opacitySlider) opacitySlider.value = 0.5;
            const opacityDisplay = document.getElementById('overlayOpacityValue');
            if (opacityDisplay) opacityDisplay.textContent = '50%';
          }
        }
        const dilateBtnE = document.getElementById('dilateBETBtn');
        if (dilateBtnE) dilateBtnE.disabled = false;
        const erodeBtnE = document.getElementById('erodeBETBtn');
        if (erodeBtnE) erodeBtnE.disabled = false;
        const applyBtnE = document.getElementById('applyBETBtn');
        if (applyBtnE) applyBtnE.disabled = false;
        break;
      }
    }

    // Load stage data into viewer for preprocessing steps
    // (stageData is already handled in handleStageData)
  }

  onVolumeInfo(info) {
    this.updateDownsampleInfo();
  }

  updateStepBadge(step, status) {
    const badgeMap = {
      'load': 'stepN4Badge', // load doesn't have its own badge, reusing
      'downsample': 'stepDownsampleBadge',
      'n4': 'stepN4Badge',
      'bet': 'stepBETBadge',
      'denoise': 'stepDenoiseBadge',
      'inference': 'stepInferenceBadge'
    };
    // Load step doesn't have a visible badge
    if (step === 'load') return;

    const badge = document.getElementById(badgeMap[step]);
    if (!badge) return;

    badge.className = 'step-badge';
    badge.textContent = '';

    switch (status) {
      case 'running':
        badge.classList.add('badge-running');
        badge.textContent = 'Running';
        break;
      case 'complete':
        badge.classList.add('badge-complete');
        badge.textContent = 'Done';
        break;
      case 'skipped':
        badge.classList.add('badge-skipped');
        badge.textContent = 'Skipped';
        break;
      case 'pending':
        badge.classList.add('badge-pending');
        badge.textContent = 'Pending';
        break;
    }
  }

  setStepEnabled(step, enabled) {
    const sectionId = this.getStepSectionId(step);
    if (!sectionId) return;

    const section = document.getElementById(sectionId);
    if (!section) return;

    if (enabled) {
      section.classList.remove('step-disabled');
    } else {
      section.classList.add('step-disabled');
    }
  }

  setStepButtonsEnabled(step, enabled) {
    const buttons = this.getStepButtonIds(step);
    if (!buttons) return;

    for (const id of buttons) {
      const btn = document.getElementById(id);
      if (btn) btn.disabled = !enabled;
    }
  }

  resetUIDownstream(fromStep) {
    const steps = ['n4', 'bet', 'denoise', 'inference'];
    const idx = steps.indexOf(fromStep);
    if (idx < 0) return;

    for (let i = idx + 1; i < steps.length; i++) {
      // BET re-run does NOT reset downstream
      if (fromStep === 'bet') break;
      this.updateStepBadge(steps[i], '');
      this.setStepEnabled(steps[i], false);
      this.setStepButtonsEnabled(steps[i], false);
    }
  }

  // ==================== Results ====================

  async handleStageData(data) {
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
      resultsSection.classList.remove('hidden');
      resultsSection.classList.remove('collapsed');
    }

    // For preprocessing stages, load as base volume in viewer
    if (data.stage !== 'segmentation') {
      const result = this.inferenceExecutor.getResult(data.stage);
      if (result?.file) {
        await this.viewerController.loadBaseVolume(result.file);
        this.currentResultTab = data.stage;
        this._inputVisible = true;
        this.applyDefaultBaseColormap();
        this.syncWindowControls();
        this.applyAutoContrast();
      }
    }

    // Rebuild the results list with all available stages
    this.rebuildResultsList();

    // Show overlay controls when segmentation arrives
    if (data.stage === 'segmentation') {
      const overlayControl = document.getElementById('overlayControl');
      if (overlayControl) overlayControl.classList.remove('hidden');
    }
  }

  rebuildResultsList() {
    const container = document.getElementById('stageButtons');
    if (!container) return;
    container.innerHTML = '';

    const stages = this.inferenceExecutor.getStageOrder();
    const dlSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>';
    const viewSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';

    // Build all rows with uniform layout: eye icon + label + download button
    const allStages = ['input', ...stages];

    for (const stage of allStages) {
      const row = document.createElement('div');
      row.className = 'volume-toggle';

      const viewBtn = document.createElement('button');
      viewBtn.className = 'view-btn';
      viewBtn.title = `View ${Config.STAGE_NAMES[stage] || stage}`;
      viewBtn.innerHTML = viewSvg;
      viewBtn.dataset.stage = stage;

      if (stage === 'segmentation') {
        // Segmentation eye toggles overlay visibility
        viewBtn.classList.toggle('active', this._segmentationVisible);
        viewBtn.addEventListener('click', () => {
          this._segmentationVisible = !this._segmentationVisible;
          this.toggleOverlayVisibility(this._segmentationVisible);
          viewBtn.classList.toggle('active', this._segmentationVisible);
        });
      } else {
        // Initialize active state based on what's currently displayed
        viewBtn.classList.toggle('active', this.currentResultTab === stage && this._inputVisible);
        // Base volume stages: toggle load/unload as base volume
        viewBtn.addEventListener('click', () => {
          if (this.currentResultTab === stage && this._inputVisible) {
            // Already showing this stage — hide it
            this._inputVisible = false;
            this.viewerController.setBaseOpacity(0);
            viewBtn.classList.remove('active');
          } else {
            this.viewStage(stage);
          }
        });
      }
      row.appendChild(viewBtn);

      const label = document.createElement('span');
      label.className = 'stage-label';
      label.textContent = Config.STAGE_NAMES[stage] || stage;
      row.appendChild(label);

      // Download button (not for input — user already has the file)
      if (stage !== 'input') {
        const dlBtn = document.createElement('button');
        dlBtn.className = 'download-btn';
        dlBtn.title = `Download ${Config.STAGE_NAMES[stage] || stage}`;
        dlBtn.innerHTML = dlSvg;
        dlBtn.addEventListener('click', () => this.inferenceExecutor.downloadStage(stage));
        row.appendChild(dlBtn);
      }

      container.appendChild(row);
    }
  }

  async viewStage(stage) {
    let file;
    if (stage === 'input') {
      file = this.inputFile;
    } else {
      const result = this.inferenceExecutor.getResult(stage);
      file = result?.file;
    }
    if (!file) return;

    await this.viewerController.loadBaseVolume(file);
    this.currentResultTab = stage;
    this._inputVisible = true;
    this.applyDefaultBaseColormap();
    this.syncWindowControls();
    this.applyAutoContrast();

    // Re-add segmentation overlay if it exists and is visible
    if (this._segmentationVisible) {
      const segResult = this.inferenceExecutor.getResult('segmentation');
      if (segResult?.file) {
        await this.viewerController.loadOverlay(segResult.file, 'red');
      }
    }

    // Update active state on view buttons
    const container = document.getElementById('stageButtons');
    if (container) {
      container.querySelectorAll('.view-btn').forEach(btn => {
        if (btn.dataset.stage === 'segmentation') {
          // Segmentation eye reflects overlay visibility, not base selection
          btn.classList.toggle('active', this._segmentationVisible);
        } else {
          btn.classList.toggle('active', btn.dataset.stage === stage);
        }
      });
    }
  }

  toggleInputVisibility(visible) {
    this._inputVisible = visible;
    this.viewerController.setBaseOpacity(visible ? 1 : 0);
  }

  toggleOverlayVisibility(visible) {
    this._segmentationVisible = visible;
    const opacitySlider = document.getElementById('overlayOpacity');
    if (visible) {
      this.viewerController.setOverlayOpacity(this._overlaySliderValue);
      if (opacitySlider) opacitySlider.disabled = false;
    } else {
      this.viewerController.setOverlayOpacity(0);
      if (opacitySlider) opacitySlider.disabled = true;
    }
    this.updateViewerInfo(this._lastLocationData);
  }

  onWorkerInitialized() {
    // If WASM preprocessing not available, auto-select SynthStrip Fast and disable BET option
    const betMethodSelect = document.getElementById('betMethodSelect');
    if (betMethodSelect && !this.inferenceExecutor.wasmAvailable) {
      betMethodSelect.value = 'synthstrip-fast';
      // Disable the BET option
      const betOption = betMethodSelect.querySelector('option[value="bet"]');
      if (betOption) betOption.disabled = true;
      // Hide FI group (not needed for SynthStrip)
      const fiGroup = document.getElementById('betFiGroup');
      if (fiGroup) fiGroup.style.display = 'none';
    }
  }

  async onInferenceComplete() {
    const statusText = document.getElementById('statusText');
    this.resetAbortControls();
    this.currentRunningStep = null;
    this.abortUICheckpoint = null;
    if (statusText) statusText.textContent = 'Ready';

    // Show segmentation overlay on top of the input volume
    const fullResult = this.inferenceExecutor.getResult('segmentation');
    const overlayFile = fullResult?.file;
    if (overlayFile) {
      // Load segmentation as overlay on top of existing base volume
      await this.viewerController.loadOverlay(overlayFile, 'red');

      // Set overlay to 50% so input remains visible underneath
      this._overlaySliderValue = 0.5;
      this.viewerController.setOverlayOpacity(0.5);
      const opacitySlider = document.getElementById('overlayOpacity');
      if (opacitySlider) opacitySlider.value = 0.5;
      const opacityDisplay = document.getElementById('overlayOpacityValue');
      if (opacityDisplay) opacityDisplay.textContent = '50%';
      this.syncWindowControls();
    }
  }

  onInferenceError(msg) {
    const statusText = document.getElementById('statusText');
    this.resetAbortControls();
    this.currentRunningStep = null;
    this.abortUICheckpoint = null;
    if (statusText) statusText.textContent = 'Error';

    // Reset any running badges back
    for (const step of Config.PIPELINE_STEPS) {
      const status = this.inferenceExecutor.getStepStatus(step);
      if (status === 'running') {
        this.updateStepBadge(step, 'pending');
        this.setStepButtonsEnabled(step, true);
      }
    }
  }

  disableAllResultTabs() {
    const container = document.getElementById('stageButtons');
    if (container) container.innerHTML = '';
    this._segmentationVisible = true;
    this._overlaySliderValue = 0.5;
  }

  clearResults() {
    this.inferenceExecutor.clearResults();
    this.disableAllResultTabs();
    this.currentResultTab = 'input';
    this._inputVisible = true;

    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
      resultsSection.classList.add('hidden');
      resultsSection.classList.add('collapsed');
    }

    const overlayControl = document.getElementById('overlayControl');
    if (overlayControl) overlayControl.classList.add('hidden');

    const applyGroup = document.getElementById('applyBETGroup');
    if (applyGroup) applyGroup.classList.add('hidden');

    const opacitySlider = document.getElementById('overlayOpacity');
    if (opacitySlider) {
      opacitySlider.disabled = false;
      opacitySlider.value = 0.5;
    }
    const opacityDisplay = document.getElementById('overlayOpacityValue');
    if (opacityDisplay) opacityDisplay.textContent = '50%';

    if (this.inputFile) {
      this.viewerController.loadBaseVolume(this.inputFile);
    }

    this.updateViewerInfo(this._lastLocationData);
  }

  // ==================== UI Helpers ====================

  updateOutput(msg) {
    this.console.log(msg);
  }

  setProgress(value, text) {
    this.progress.setProgress(value);
    const statusText = document.getElementById('statusText');
    if (statusText) {
      if (value >= 1) statusText.textContent = 'Complete';
      else if (text) statusText.textContent = text;
      else if (value > 0) statusText.textContent = 'Processing...';
    }
  }

  clearFiles() {
    this.fileIOController.clearFiles();
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.app = new VesselBoostApp();
});
