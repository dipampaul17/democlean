# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-02

### Added
- **Embedding backends**: New `--encoder` flag to choose between `raw` (default) and `hpt`
- **HPT embeddings**: Pre-trained embeddings that reduce length correlation from 0.80 to 0.32
- `--hpt-model` flag to select HPT model size (small/base/large)
- `--device` flag for GPU acceleration with HPT
- `get_encoder()` factory function in Python API
- `Encoder` base class for custom embedding backends
- Validation script (`scripts/validate_hpt.py`)

### Changed
- `DemoScorer` now accepts optional `encoder` parameter
- `EpisodeScore.metadata` includes encoder name
- `EpisodeScore.state_dim` now reports original input dimensions

### Technical
- New `democlean.embeddings` module with abstract `Encoder` base class
- Optional `[hpt]` dependency group for PyTorch

## [0.1.5] - 2026-02-02

### Fixed
- CI workflow

## [0.1.4] - 2026-02-02

### Changed
- Cleaner README formatting

## [0.1.3] - 2026-02-02

### Changed
- README polish

## [0.1.2] - 2026-02-02

### Changed
- Added DemInf citation to credits

## [0.1.1] - 2026-02-02

### Changed
- Improved README clarity and documentation

## [0.1.0] - 2026-02-02

### Added
- Initial release
- KSG mutual information scoring for LeRobot episodes
- CLI with `analyze` command
- JSON report output
- Length normalization option (`--normalize-length`)
- MI threshold filtering (`--min-mi`)
- MI variance warnings for uniform quality datasets
- Bootstrap confidence intervals (`--ci`)
- PCA dimension reduction for high-dimensional states (`--max-dim`)
- Merge with score_lerobot_episodes output (`--merge`)
- Python API with `DemoScorer` class
