# democlean

[![PyPI](https://img.shields.io/pypi/v/democlean.svg)](https://pypi.org/project/democlean/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Score robot demonstrations by motion quality.

## The Problem

Robot learning datasets contain bad demonstrations—jerky movements, hesitation, inconsistent timing. Training on these hurts policy performance. Manual review doesn't scale.

## The Solution

democlean scores episodes using mutual information (MI) between states and actions. High MI means smooth, purposeful motion. Low MI means noisy or hesitant behavior.

## Install

```bash
pip install democlean
```

## Quick Start

```bash
democlean analyze lerobot/pusht
```

```
Dataset lerobot/pusht
Episodes: 206 | Dims: 2→2

Distribution
  ████████████████████  High   124
  ██████████            Medium  62
  █████                 Low     20

Mean   2.55   (typical for human teleop)

Flagged (lowest MI)
  ep  46  1.897
  ep   6  1.984
```

### Filter Bad Episodes

```bash
democlean analyze lerobot/pusht --keep 0.8      # keep top 80%
democlean analyze lerobot/pusht --min-mi 2.0    # drop below threshold
democlean analyze lerobot/pusht -r report.json  # save results
```

## Interpreting MI Scores

| MI Score | Quality | Typical Source |
|----------|---------|----------------|
| > 3.0 | Excellent | Clean scripted demos |
| 2.0 – 3.0 | Good | Skilled human teleop |
| 1.0 – 2.0 | Moderate | Novice operators, review recommended |
| < 1.0 | Poor | Random/broken collection |

MI measures *how* the robot moved, not *what* it achieved. Use task-specific metrics for success rates.

## Embedding Backends

### Raw (default)

Computes MI directly on state/action features.

```bash
democlean analyze lerobot/pusht --encoder raw
```

**Limitation:** MI correlates with episode length (r ≈ 0.8). Longer episodes score higher regardless of quality. Use `--normalize-length` to partially adjust.

### HPT (experimental)

Embeds states/actions using a neural network before computing MI. Based on the [HPT](https://github.com/liruiw/HPT) architecture.

```bash
pip install democlean[hpt]
democlean analyze lerobot/pusht --encoder hpt
```

**Benefit:** Reduces length correlation from 0.80 to 0.32.
**Cost:** Requires PyTorch, ~2x slower.

| Encoder | Length Correlation | Speed |
|---------|-------------------|-------|
| raw | 0.80 | ~35 eps/sec |
| hpt | 0.32 | ~20 eps/sec |

Use HPT when episode lengths vary significantly and you want quality scores less dominated by length.

## Python API

```python
from democlean import DemoScorer, get_encoder

# Basic usage
scorer = DemoScorer(k=3)
scores = scorer.score_dataset("lerobot/pusht")
keep = scorer.filter_top_k(scores, percentile=80)

# With HPT embeddings
encoder = get_encoder("hpt", model_size="base")
scorer = DemoScorer(k=3, encoder=encoder)
```

## CLI Reference

| Flag | Description |
|------|-------------|
| `--keep R` | Keep top fraction (0–1) |
| `--top-k K` | Keep top K episodes |
| `--min-mi T` | Drop episodes below MI threshold |
| `--normalize-length` | Divide MI by log(length) |
| `--encoder {raw,hpt}` | Embedding backend |
| `--hpt-model {small,base,large}` | HPT model size |
| `-k N` | KSG nearest neighbors (default: 3) |
| `--max-dim D` | PCA reduce state dimensions |
| `--ci` | Compute bootstrap confidence intervals |
| `-r FILE` | Save JSON report |
| `-q` | Quiet mode (JSON output only) |

## When to Use

**Good fit:**
- Human teleoperation data with quality variation
- 50+ episodes for reliable statistics
- Quick triage before expensive training runs

**Not ideal:**
- Scripted simulation data (already uniform quality)
- Multi-task datasets (MI varies by task complexity)
- When you need task success metrics

## Limitations

1. **Not task success.** MI measures motion consistency, not whether the task was completed.
2. **Length correlation.** Raw MI correlates with episode length. Use HPT encoder or `--normalize-length`.
3. **Sample size.** Statistics are unreliable below ~20 episodes.

## Credits

Based on ideas from [Demonstration Information](https://arxiv.org/abs/2502.08623) (Hejna et al.).

Pairs well with [score_lerobot_episodes](https://github.com/RoboticsData/score_lerobot_episodes) for visual quality metrics.

## License

MIT
