/**
 * Modal Manager Module
 * Simple utility to manage modal open/close state.
 */
export class ModalManager {
  constructor(modalId) {
    this.modalId = modalId;
    this.modal = document.getElementById(modalId);
    if (this.modal) {
      this._setupOverlayClickClose();
    }
  }

  open() {
    if (this.modal) this.modal.classList.add('active');
  }

  close() {
    if (this.modal) this.modal.classList.remove('active');
  }

  isOpen() {
    return this.modal?.classList.contains('active') ?? false;
  }

  toggle() {
    if (this.isOpen()) this.close();
    else this.open();
  }

  _setupOverlayClickClose() {
    this.modal.addEventListener('click', (e) => {
      if (e.target === this.modal) this.close();
    });
  }
}
