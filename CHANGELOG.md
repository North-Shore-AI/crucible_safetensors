# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-20

### Added
- Added metadata-only `CrucibleSafetensors.Manifest` inventory and validation for
  expected tensor names, dtypes, shapes, byte sizes, and exact inventory checks.
- Added streaming SHA-256 calculation plus explicit digest verification helpers.
- Centralized SafeTensors dtype normalization and bit-width metadata in
  `CrucibleSafetensors.TensorInfo`, including current byte and sub-byte format dtypes.
- Added header-size, duplicate-key, contiguous-layout, byte-alignment, and metadata-format
  validation for safer artifact preflight.

### Changed
- Made the package tensor-runtime-neutral: runtime dependencies are now only file-format
  dependencies and `jason`; Nx is no longer forced or overridden.
- Reader, writer, vector inspection, and manifest validation now share one dtype vocabulary.
- Writer payload validation supports the expanded raw SafeTensors dtype vocabulary.

### Removed
- Removed the legacy `Crucible.Safetensors.Slice` `%Safetensors.FileTensor{}` / Nx bridge.
  Use `CrucibleSafetensors.Reader`, `ChunkReader`, and the consuming runtime's own tensor
  materialization instead.
- Removed direct `:nx` and external `:safetensors` dependencies.

## [0.1.0] - 2026-05-23

### Added
- Initial release of `CrucibleSafetensors` from the monolith extraction.
- Added metadata-only SafeTensors vector inspection for candidate preflight
  checks.
