/**
 * FileIOController
 *
 * Handles unified file input for vessel segmentation.
 * Auto-detects NIfTI vs DICOM and converts as needed.
 */

import { DicomController } from './DicomController.js';

export class FileIOController {
  constructor(options) {
    this.updateOutput = options.updateOutput || (() => {});
    this.onFileLoaded = options.onFileLoaded || (() => {});

    this.file = null;

    this.dicomController = new DicomController({
      updateOutput: (msg) => this.updateOutput(msg),
      onConversionComplete: (niftiFile) => {
        this.file = niftiFile;
        this._updateUI(niftiFile.name);
        this.onFileLoaded(niftiFile);
        this._resetNativeInput();
      }
    });
  }

  getActiveFile() {
    return this.file;
  }

  hasValidData() {
    return this.file !== null;
  }

  handleFiles(files) {
    if (!files || files.length === 0) return;

    if (this._isNifti(files)) {
      const niftiFile = this._findNiftiFile(files);
      this.file = niftiFile;
      this._updateUI(niftiFile.name);
      this.updateOutput(`Loaded: ${niftiFile.name}`);
      this.onFileLoaded(niftiFile);
      this._resetNativeInput();
    } else {
      this.updateOutput(`Detected DICOM input (${files.length} files)`);
      this.dicomController.convertFiles(Array.from(files));
    }
  }

  handleDropItems(dataTransferItems) {
    if (!dataTransferItems || dataTransferItems.length === 0) return;

    // Check if any dropped item is a NIfTI file
    const files = [];
    for (let i = 0; i < dataTransferItems.length; i++) {
      const file = dataTransferItems[i].getAsFile?.();
      if (file) files.push(file);
    }

    if (files.length > 0 && this._isNifti(files)) {
      this.handleFiles(files);
      return;
    }

    // Otherwise treat as DICOM (may be folder drop)
    this.dicomController.convertDropItems(dataTransferItems);
  }

  _isNifti(files) {
    return Array.from(files).some(f => {
      const name = f.name.toLowerCase();
      return name.endsWith('.nii') || name.endsWith('.nii.gz');
    });
  }

  _findNiftiFile(files) {
    return Array.from(files).find(f => {
      const name = f.name.toLowerCase();
      return name.endsWith('.nii') || name.endsWith('.nii.gz');
    });
  }

  _updateUI(filename) {
    const dropZone = document.getElementById('inputDropZone');
    const fileList = document.getElementById('fileList');

    if (dropZone) {
      dropZone.classList.add('has-files');
      const label = dropZone.querySelector('.file-drop-label span');
      if (label) label.textContent = filename;
    }

    if (fileList) {
      fileList.innerHTML = '';
      const fileItem = document.createElement('div');
      fileItem.className = 'file-item';
      fileItem.innerHTML = `
        <span>${filename}</span>
        <button class="file-remove" onclick="app.clearFiles()">&times;</button>
      `;
      fileList.appendChild(fileItem);
    }
  }

  clearFiles() {
    this.file = null;
    const dropZone = document.getElementById('inputDropZone');
    const fileList = document.getElementById('fileList');

    if (dropZone) {
      dropZone.classList.remove('has-files');
      const label = dropZone.querySelector('.file-drop-label span');
      if (label) label.textContent = 'Drop NIfTI or DICOM files';
    }
    if (fileList) fileList.innerHTML = '';
    this._resetNativeInput();
  }

  _resetNativeInput() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.value = '';
  }
}
