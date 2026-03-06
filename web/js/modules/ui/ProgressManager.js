/**
 * Progress Manager Module
 * Handles progress bar updates with smooth animation.
 */
export class ProgressManager {
  constructor(config) {
    this.progressBarId = 'progressBar';
    this.progress = 0;
    this.targetProgress = 0;
    this.animatedProgress = 0;
    this.progressAnimationId = null;
    this.lastAnimationTime = 0;
    this.animationSpeed = config.animationSpeed;
  }

  setProgress(value, text = null) {
    this.progress = value;
    this.targetProgress = value;
    this.animatedProgress = value;
    this.updateProgressBar();
    this.stopAnimation();
  }

  animate() {
    const now = performance.now();
    const deltaTime = (now - this.lastAnimationTime) / 1000;
    this.lastAnimationTime = now;
    if (this.animatedProgress < this.targetProgress) {
      const increment = this.animationSpeed * deltaTime;
      this.animatedProgress = Math.min(this.animatedProgress + increment, this.targetProgress);
      this.updateProgressBar();
    }
    if (this.targetProgress < 1 && this.targetProgress > 0) {
      this.progressAnimationId = requestAnimationFrame(() => this.animate());
    }
  }

  startAnimation() {
    if (this.progressAnimationId) return;
    this.lastAnimationTime = performance.now();
    this.progressAnimationId = requestAnimationFrame(() => this.animate());
  }

  stopAnimation() {
    if (this.progressAnimationId) {
      cancelAnimationFrame(this.progressAnimationId);
      this.progressAnimationId = null;
    }
  }

  updateProgressBar() {
    const bar = document.getElementById(this.progressBarId);
    if (bar) bar.style.width = `${this.animatedProgress * 100}%`;
  }

  reset() {
    this.setProgress(0);
  }
}
