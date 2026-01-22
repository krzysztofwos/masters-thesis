// Auto-fit slide body content for reveal.js slides
// Dynamically scales content to prevent vertical overflow
// Pre-computes scales to avoid visible resize on slide change
(function () {
  "use strict";

  const CONFIG = {
    debug: false,
    minScale: 0.55, // Don't scale below 55% of original size
    maxScale: 1.0, // Never scale up
    safetyMargin: 0.9, // Leave 10% margin for safety
    overflowThreshold: 10, // Pixels of overflow tolerance
    debounceDelay: 100, // Ms to wait before re-fitting on resize
    legendPadding: 6, // Pixels to keep above legend
  };

  // Cache for pre-computed scales: Map<slideElement, {scale: number, baseFontSize: number}>
  const scaleCache = new WeakMap();

  function log(...args) {
    if (CONFIG.debug) console.log("[fit-body]", ...args);
  }

  /**
   * Check if a slide has content that could overflow
   */
  function hasSignificantContent(slide) {
    return slide.querySelector(
      "p, ul, ol, .columns, blockquote, table, pre, .MathJax",
    );
  }

  /**
   * Check if slide has overflow using scrollHeight
   */
  function hasOverflow(slide) {
    if (slide.scrollHeight > slide.clientHeight + CONFIG.overflowThreshold) {
      return true;
    }

    const legend = slide.querySelector(".legend");
    if (!legend) return false;

    const legendRect = legend.getBoundingClientRect();
    if (legendRect.height === 0 || legendRect.width === 0) return false;

    const slideRect = slide.getBoundingClientRect();
    const legendTop = legendRect.top - slideRect.top;

    let maxBottom = 0;
    Array.from(slide.children).forEach((child) => {
      if (child === legend) return;
      if (legend && child.contains(legend)) return;
      if (child.matches(".notes, aside.notes")) return;

      const rect = child.getBoundingClientRect();
      if (rect.height === 0 || rect.width === 0) return;

      const bottom = rect.bottom - slideRect.top;
      if (bottom > maxBottom) maxBottom = bottom;
    });

    return maxBottom > legendTop - CONFIG.legendPadding;
  }

  /**
   * Reset a slide's font size to default
   */
  function resetSlide(slide) {
    slide.style.fontSize = "";
    slide.removeAttribute("data-fitted");
    slide.removeAttribute("data-fit-scale");
  }

  /**
   * Apply cached scale to a slide (instant, no recalculation)
   */
  function applyScale(slide) {
    const cached = scaleCache.get(slide);
    if (cached && cached.scale < 1.0) {
      slide.style.fontSize = cached.baseFontSize * cached.scale + "px";
      slide.setAttribute("data-fitted", "true");
      slide.setAttribute("data-fit-scale", cached.scale.toFixed(2));
    }
  }

  /**
   * Calculate and cache the optimal scale for a slide
   */
  function calculateScale(slide) {
    if (!slide || slide.classList.contains("stack")) return;
    if (!hasSignificantContent(slide)) return;

    // Reset first to get accurate measurements
    resetSlide(slide);
    void slide.offsetHeight;

    // Get base font size
    const computedStyle = window.getComputedStyle(slide);
    const baseFontSize = parseFloat(computedStyle.fontSize);

    if (!hasOverflow(slide)) {
      // No overflow, cache scale of 1.0
      scaleCache.set(slide, { scale: 1.0, baseFontSize });
      log(
        `Slide "${slide.querySelector("h2")?.textContent?.substring(0, 30) || "untitled"}" - no overflow`,
      );
      return;
    }

    log(
      `Slide "${slide.querySelector("h2")?.textContent?.substring(0, 30) || "untitled"}" has overflow, calculating...`,
    );

    // Binary search for the right scale
    let minScale = CONFIG.minScale;
    let maxScale = CONFIG.maxScale;
    let bestScale = minScale;

    for (let i = 0; i < 10; i++) {
      const midScale = (minScale + maxScale) / 2;
      slide.style.fontSize = baseFontSize * midScale + "px";
      void slide.offsetHeight;

      if (hasOverflow(slide)) {
        maxScale = midScale;
      } else {
        bestScale = midScale;
        minScale = midScale;
      }

      if (maxScale - minScale < 0.01) break;
    }

    // Apply safety margin and cache
    const finalScale = Math.max(
      CONFIG.minScale,
      bestScale * CONFIG.safetyMargin,
    );
    scaleCache.set(slide, { scale: finalScale, baseFontSize });

    log(`Calculated scale ${finalScale.toFixed(2)} for slide`);
  }

  /**
   * Get all content slides
   */
  function getAllSlides() {
    return document.querySelectorAll(
      ".reveal .slides > section:not(.stack), " +
        ".reveal .slides > section.stack > section",
    );
  }

  /**
   * Pre-compute scales for all slides (called once at init and on resize)
   */
  function precomputeAllScales() {
    log("Pre-computing scales for all slides...");
    const slides = getAllSlides();

    // Reset all slides first
    slides.forEach(resetSlide);
    void document.body.offsetHeight;

    // Calculate scale for each slide
    slides.forEach(calculateScale);

    // Apply scales to all slides
    slides.forEach(applyScale);

    log(`Pre-computed scales for ${slides.length} slides`);
  }

  /**
   * Debounced version for resize
   */
  let resizeTimeout;
  function debouncedPrecompute() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      // Clear cache on resize since dimensions changed
      precomputeAllScales();
    }, CONFIG.debounceDelay);
  }

  /**
   * Initialize - pre-compute all scales
   */
  function init() {
    log("Initializing fit-body.js");

    // Hide slides container briefly during computation to prevent flash
    const slidesContainer = document.querySelector(".reveal .slides");
    if (slidesContainer) {
      slidesContainer.style.opacity = "0";
    }

    // Pre-compute all scales
    setTimeout(() => {
      precomputeAllScales();

      // Restore visibility
      if (slidesContainer) {
        slidesContainer.style.opacity = "";
      }
    }, 100);
  }

  // Integration with Reveal.js
  if (window.Reveal) {
    Reveal.on("ready", init);

    // On slide change, apply cached scale and verify visible slide fits
    Reveal.on("slidechanged", (event) => {
      const slide = event.currentSlide;
      applyScale(slide);
      if (hasOverflow(slide)) {
        calculateScale(slide);
        applyScale(slide);
      }
    });

    Reveal.on("resize", debouncedPrecompute);
  } else {
    document.addEventListener("DOMContentLoaded", () => {
      setTimeout(init, 500);
    });
  }

  // Recompute after assets and fonts load
  window.addEventListener("load", debouncedPrecompute);
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(debouncedPrecompute);
  }

  // Handle window resize
  window.addEventListener("resize", debouncedPrecompute);

  // Expose debugging API
  window.fitBody = {
    precompute: precomputeAllScales,
    applyScale: applyScale,
    config: CONFIG,
    enableDebug: () => {
      CONFIG.debug = true;
      console.log("fit-body debug enabled");
    },
    disableDebug: () => {
      CONFIG.debug = false;
    },

    /**
     * Get cached scales for all slides
     */
    getCachedScales: () => {
      const slides = getAllSlides();
      const result = [];
      slides.forEach((slide, index) => {
        const cached = scaleCache.get(slide);
        if (cached) {
          result.push({
            index,
            title:
              slide.querySelector("h2")?.textContent?.substring(0, 40) ||
              `Slide ${index}`,
            scale: cached.scale.toFixed(2),
            fitted: cached.scale < 1.0,
          });
        }
      });
      console.table(result.filter((r) => r.fitted));
      return result;
    },

    /**
     * Check current overflow status
     */
    checkOverflow: () => {
      const slides = getAllSlides();
      const overflowing = [];
      slides.forEach((slide, index) => {
        if (hasOverflow(slide)) {
          overflowing.push({
            index,
            title:
              slide.querySelector("h2")?.textContent?.substring(0, 40) ||
              `Slide ${index}`,
            overflow: slide.scrollHeight - slide.clientHeight,
          });
        }
      });
      console.log("Slides with overflow:", overflowing.length);
      if (overflowing.length > 0) console.table(overflowing);
      return overflowing;
    },
  };

  log("fit-body.js loaded. Use window.fitBody for debugging.");
})();
