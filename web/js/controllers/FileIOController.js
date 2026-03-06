/**
 * FileIOController
 *
 * Handles single MRI file input for vessel segmentation.
 * Supports both NIfTI and DICOM input modes.
 */

export class FileIOController {
  constructor(options) {
    this.updateOutput = options.updateOutput || (() => {});
    this.onFileLoaded = options.onFileLoaded || (() => {});

    this.inputMode = 'dicom';
    this.niftiFile = null;
    this.dicomFile = null;
  }

  getInputMode() { return this.inputMode; }
  setInputMode(mode) { this.inputMode = mode; }

  getActiveFile() {
    return this.inputMode === 'dicom' ? this.dicomFile : this.niftiFile;
  }

  hasValidData() {
    return this.getActiveFile() !== null;
  }

  handleFileInput(event) {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    this.niftiFile = files[0];
    this.updateFileListUI('nifti', [this.niftiFile]);
    this.updateOutput(`Loaded: ${this.niftiFile.name}`);
    this.onFileLoaded(this.niftiFile);
  }

  setFileFromDicom(file) {
    this.dicomFile = file;
    this.updateFileListUI('dicom', [file]);
    this.onFileLoaded(file);
  }

  updateFileListUI(type, files) {
    const listElement = document.getElementById(`${type}List`);
    const fileDrop = listElement?.closest('.upload-group')?.querySelector('.file-drop');

    if (!listElement) return;

    listElement.innerHTML = '';

    if (files && files.length > 0) {
      fileDrop?.classList.add('has-files');
      files.forEach((file) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
          <span>${file.name}</span>
          <button class="file-remove" onclick="app.clearFiles('${type}')">&times;</button>
        `;
        listElement.appendChild(fileItem);
      });

      const label = fileDrop?.querySelector('.file-drop-label span');
      if (label) label.textContent = files[0].name || '1 file selected';
    } else {
      fileDrop?.classList.remove('has-files');
      const label = fileDrop?.querySelector('.file-drop-label span');
      if (label) label.textContent = 'Drop or click';
    }
  }

  clearFiles(type) {
    if (type === 'nifti') {
      this.niftiFile = null;
    } else if (type === 'dicom') {
      this.dicomFile = null;
    }
    this.updateFileListUI(type, []);
  }

  clearAllFiles() {
    this.niftiFile = null;
    this.dicomFile = null;
    this.updateFileListUI('nifti', []);
    this.updateFileListUI('dicom', []);
  }
}
