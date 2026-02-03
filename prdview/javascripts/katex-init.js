/* global renderMathInElement */

function docviewRenderKatex() {
  if (typeof window.renderMathInElement !== "function") return;
  try {
    window.renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  } catch (_) {
    // ignore
  }
}

if (window.document$ && typeof window.document$.subscribe === "function") {
  window.document$.subscribe(docviewRenderKatex);
} else {
  document.addEventListener("DOMContentLoaded", docviewRenderKatex);
}
