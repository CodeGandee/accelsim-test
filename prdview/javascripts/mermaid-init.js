/* global mermaid, document$ */

function docviewRenderMermaid() {
  if (!window.mermaid) return;

  try {
    window.mermaid.initialize({ startOnLoad: false });
  } catch (_) {
    // ignore
  }

  try {
    const maybePromise = window.mermaid.run({ querySelector: ".mermaid" });
    if (maybePromise && typeof maybePromise.catch === "function") {
      maybePromise.catch(() => undefined);
    }
  } catch (_) {
    // ignore
  }
}

if (window.document$ && typeof window.document$.subscribe === "function") {
  window.document$.subscribe(docviewRenderMermaid);
} else {
  document.addEventListener("DOMContentLoaded", docviewRenderMermaid);
}
