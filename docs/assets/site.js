"use strict";

const copyButton = document.getElementById("copy-citation");
const citation = document.getElementById("bibtex");
const copyStatus = document.getElementById("copy-status");
let resetTimer;

if (copyButton && citation && copyStatus) {
  copyButton.hidden = false;
  copyButton.addEventListener("click", async () => {
    clearTimeout(resetTimer);
    try {
      if (!navigator.clipboard || !window.isSecureContext) {
        throw new Error("Clipboard API unavailable");
      }
      await navigator.clipboard.writeText(citation.textContent.trim() + "\n");
      copyButton.querySelector("span").textContent = "Copied!";
      copyStatus.textContent = "BibTeX copied to clipboard.";
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(citation);
      selection.removeAllRanges();
      selection.addRange(range);
      copyButton.querySelector("span").textContent = "Citation selected";
      copyStatus.textContent =
        "Press Ctrl+C (Windows) or ⌘C (Mac) to copy the selected citation.";
    }
    resetTimer = window.setTimeout(() => {
      copyButton.querySelector("span").textContent = "Copy citation";
      copyStatus.textContent = "";
    }, 5000);
  });
}
