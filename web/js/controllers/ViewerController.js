/**
 * ViewerController
 *
 * Manages NiiVue visualization with support for base volume and segmentation overlays.
 * Adapted for VesselBoost's binary vessel colormap.
 */

export class ViewerController {
  constructor(options) {
    this.nv = options.nv;
    this.updateOutput = options.updateOutput || (() => {});
    this.currentBaseFile = null;
    this.currentOverlayFile = null;
    this.vesselColormapRegistered = false;
  }

  /**
   * Register the VesselBoost discrete colormap with NiiVue.
   * @param {Object} colormapData - { R, G, B, A } arrays from labels.js
   */
  registerVesselColormap(colormapData) {
    try {
      this.nv.addColormap('vesselboost', colormapData);
      this.vesselColormapRegistered = true;
    } catch (e) {
      console.warn('Could not register vesselboost colormap:', e);
    }
  }

  async loadBaseVolume(file) {
    try {
      this.updateOutput(`Loading ${file.name}...`);
      const url = URL.createObjectURL(file);
      await this.nv.loadVolumes([{ url: url, name: file.name }]);
      URL.revokeObjectURL(url);
      this.currentBaseFile = file;
      this.currentOverlayFile = null;
      this.updateOutput(`${file.name} loaded`);
    } catch (error) {
      this.updateOutput(`Error loading ${file.name}: ${error.message}`);
      console.error(error);
    }
  }

  async loadOverlay(file, colormap = 'red', opacity = 0.5) {
    try {
      const url = URL.createObjectURL(file);
      await this.nv.addVolumeFromUrl({ url: url, name: file.name, colormap: colormap });
      URL.revokeObjectURL(url);

      if (this.nv.volumes.length > 1) {
        // Binary mask: cal_min=0, cal_max=1
        this.nv.volumes[1].cal_min = 0;
        this.nv.volumes[1].cal_max = 1;
        this.nv.setOpacity(1, opacity);
        this.nv.updateGLVolume();
      }

      this.currentOverlayFile = file;
    } catch (error) {
      this.updateOutput(`Error loading overlay: ${error.message}`);
      console.error(error);
    }
  }

  async showResultAsOverlay(baseFile, overlayFile, colormap = 'vesselboost') {
    await this.loadBaseVolume(baseFile);
    if (overlayFile) {
      await this.loadOverlay(overlayFile, colormap);
    }
  }

  setViewType(type) {
    const typeMap = {
      multiplanar: this.nv.sliceTypeMultiplanar,
      axial: this.nv.sliceTypeAxial,
      coronal: this.nv.sliceTypeCoronal,
      sagittal: this.nv.sliceTypeSagittal,
      render: this.nv.sliceTypeRender
    };
    if (typeMap[type] !== undefined) {
      this.nv.setSliceType(typeMap[type]);
    }
  }

  setBaseOpacity(value) {
    if (this.nv.volumes.length > 0) {
      this.nv.setOpacity(0, value);
      this.nv.updateGLVolume();
    }
  }

  setOverlayOpacity(value) {
    if (this.nv.volumes.length > 1) {
      this.nv.setOpacity(1, value);
      this.nv.updateGLVolume();
    }
  }

  setOverlayColormap(colormap) {
    if (this.nv.volumes.length > 1) {
      this.nv.volumes[1].colormap = colormap;
      this.nv.updateGLVolume();
    }
  }

  getCurrentFile() {
    return this.currentBaseFile;
  }
}
