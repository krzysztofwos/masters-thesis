// Auto-fit text for reveal.js slides
(function () {
  function fitText() {
    // Select all h2 elements in reveal.js slides
    const fitElements = document.querySelectorAll(
      ".reveal .slides section > h2",
    );

    fitElements.forEach((element) => {
      // Reset to default to get accurate measurements
      element.style.fontSize = "";

      const slide = element.closest("section");
      if (!slide) return;

      // Get the original font size as the maximum
      const computedStyle = window.getComputedStyle(element);
      const originalSize = parseFloat(computedStyle.fontSize);

      // Get available width (account for reveal.js)
      const availableWidth = slide.offsetWidth * 0.95;

      // Create a temporary span for accurate measurement
      const measureSpan = document.createElement("span");
      measureSpan.style.visibility = "hidden";
      measureSpan.style.position = "absolute";
      measureSpan.style.whiteSpace = "nowrap";
      measureSpan.style.fontFamily = computedStyle.fontFamily;
      measureSpan.style.fontWeight = computedStyle.fontWeight;
      measureSpan.textContent = element.textContent;
      document.body.appendChild(measureSpan);

      // Binary search for optimal font size (capped at original size)
      let minSize = 10;
      let maxSize = Math.min(originalSize, 200); // Cap at original size
      let bestSize = minSize;

      while (minSize <= maxSize) {
        const midSize = Math.floor((minSize + maxSize) / 2);
        measureSpan.style.fontSize = midSize + "px";

        if (measureSpan.offsetWidth <= availableWidth) {
          bestSize = midSize;
          minSize = midSize + 1;
        } else {
          maxSize = midSize - 1;
        }
      }

      // Clean up
      document.body.removeChild(measureSpan);

      // Apply the calculated font size
      element.style.fontSize = bestSize + "px";
    });
  }

  // Wait for Reveal.js to be fully loaded
  if (window.Reveal) {
    Reveal.on("ready", () => {
      setTimeout(fitText, 100);
    });
    Reveal.on("slidechanged", fitText);
    Reveal.on("resize", fitText);
  } else {
    // Fallback if Reveal isn't loaded yet
    document.addEventListener("DOMContentLoaded", () => {
      setTimeout(fitText, 500);
    });
  }

  // Handle window resize
  window.addEventListener("resize", () => {
    setTimeout(fitText, 100);
  });
})();
