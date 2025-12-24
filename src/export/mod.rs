//! Export functionality for analysis and research
//!
//! This module provides functionality to export game analysis data in various formats.
//! Currently supports CSV export of Active Inference decompositions.

mod aif_csv;

pub use aif_csv::{AifCsvExporter, AifExportConfig, AifExportRecord};
