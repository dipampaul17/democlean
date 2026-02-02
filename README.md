# democlean

[![PyPI](https://img.shields.io/pypi/v/democlean.svg)](https://pypi.org/project/democlean/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Quality scoring for robot demonstration datasets.

## Why

Robot learning datasets contain bad demonstrations—jerky movements, hesitation, inconsistent timing. Training on these hurts performance. Manual review doesn't scale.

democlean scores episodes by motion quality using mutual information (MI) between states and actions.

## Install

```bash
pip install democlean
```

## Usage

```bash
democlean analyze lerobot/pusht
```

```
Dataset lerobot/pusht
Episodes: 50 | Dims: 2→2

Distribution
  ████████████████████  High   30
  ██████████            Medium 15
  █████                 Low     5

Mean   2.55   (typical for human teleop)

Flagged (lowest MI)
  ep  46  1.897
  ep   6  1.984
```

### Filtering

```bash
democlean analyze lerobot/pusht --keep 0.8           # keep top 80%
democlean analyze lerobot/pusht --min-mi 2.0         # drop below threshold
democlean analyze lerobot/pusht --keep 0.8 -r out.json
```

## What MI Measures

MI quantifies how predictable actions are given states.

- **High MI** → smooth, purposeful motion
- **Low MI** → jerky, hesitant, inconsistent

MI measures *how* the robot moved, not *what* it achieved. Use task-specific metrics for success rates.

| MI | Interpretation |
|----|----------------|
| >3.0 | Very smooth |
| 2.0–3.0 | Typical human teleop |
| 1.0–2.0 | Moderate |
| <1.0 | Noisy/random |

## When to Use

**Good fit:**
- Human teleoperation data
- 50+ episodes
- Quick triage before training

**Not a good fit:**
- Scripted simulation (already uniform)
- Multi-task datasets
- Need task success metrics

## Limitations

1. **Length correlation** — MI correlates with episode length (r≈0.8). Use `--normalize-length` to adjust.
2. **Not task success** — Measures motion quality, not task completion.
3. **Sample size** — Works best with 50+ episodes.

## Python API

```python
from democlean import DemoScorer

scorer = DemoScorer(k=3)
scores = scorer.score_dataset("lerobot/pusht")
keep = scorer.filter_top_k(scores, percentile=80)
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--keep R` | Keep top R fraction (0–1) |
| `--top-k K` | Keep top K episodes |
| `--min-mi T` | Drop below threshold |
| `--normalize-length` | Adjust for episode length |
| `-k N` | KSG neighbors (default: 3) |
| `--max-dim D` | PCA reduce dimensions |
| `--ci` | Bootstrap confidence intervals |
| `-r FILE` | Save JSON report |
| `-q` | Quiet mode |

## Credits

Inspired by [DemInf](https://arxiv.org/abs/2502.08623) (Hejna et al., RSS 2025).

Complements [score_lerobot_episodes](https://github.com/RoboticsData/score_lerobot_episodes) for visual quality.

## License

MIT
