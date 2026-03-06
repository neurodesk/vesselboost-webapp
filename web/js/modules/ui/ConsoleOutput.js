/**
 * Console Output Module
 * Handles logging output to the UI console panel.
 */
export class ConsoleOutput {
  constructor(outputElementId = 'consoleOutput') {
    this.outputElementId = outputElementId;
  }

  log(text) {
    const outputElement = document.getElementById(this.outputElementId);
    if (outputElement) {
      const time = new Date().toLocaleTimeString('en-US', { hour12: false });
      const line = document.createElement('div');
      line.className = 'console-line';
      line.innerHTML = `<span class="console-time">[${time}]</span> <span class="console-message">${text}</span>`;
      outputElement.appendChild(line);
      outputElement.scrollTop = outputElement.scrollHeight;
    }
    console.log(text);
  }

  copyToClipboard() {
    const outputElement = document.getElementById(this.outputElementId);
    if (!outputElement) return;
    const lines = outputElement.querySelectorAll('.console-line');
    const text = Array.from(lines).map(line => line.textContent).join('\n');
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.getElementById('copyConsole');
      if (btn) {
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
      }
    });
  }

  clear() {
    const outputElement = document.getElementById(this.outputElementId);
    if (outputElement) {
      outputElement.innerHTML = '';
    }
  }
}
