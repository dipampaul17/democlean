# democlean

Score robot demonstrations by motion quality. Identify jerky, inconsistent episodes before training.

## Install

```bash
pip install democlean
```

## Usage

```bash
# Score a dataset
democlean analyze lerobot/pusht

# Keep top 80%
democlean analyze lerobot/pusht --keep 0.8 -r report.json

# Filter by threshold
democlean analyze lerobot/pusht --min-mi 2.0

# Normalize by episode length
democlean analyze lerobot/pusht --normalize-length --keep 0.8
```

## Output

```
democlean 0.1.0

Dataset lerobot/pusht
Episodes: 50 | Dims: 2->2

Distribution
  ███████████████████████████████  High   30
  ██████████                       Medium 15
  █████                            Low     5

Mean   2.55   (typical for human teleop)
Std    0.27
Range  [1.90, 3.04]

Flagged (lowest MI)
  ep  46  1.897
  ep   6  1.984
```

## How It Works

democlean computes Mutual Information (MI) between states and actions using the KSG estimator.

**High MI indicates:**
- Temporally smooth actions
- Low jerk motion
- Purposeful movement

**Low MI indicates:**
- Jerky, hesitant motion
- Inconsistent action timing

**Limitations:**
- MI correlates with episode length (r~0.8). Use `--normalize-length` to account for this.
- Does not detect task failure. Use task-specific metrics for that.
- Works best with 50+ episodes.

## Interpreting Scores

| MI Range | Meaning |
|----------|---------|
| > 3.0 | Very smooth, consistent |
| 2.0 - 3.0 | Typical human teleop |
| 1.0 - 2.0 | Moderate quality |
| < 1.0 | Noisy or random |

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
| `--keep R` | Keep top R ratio (0-1) |
| `--top-k K` | Keep top K episodes |
| `--min-mi T` | Drop episodes below threshold T |
| `--normalize-length` | Normalize MI by episode length |
| `-k N` | KSG neighbors (default: 3) |
| `--max-dim D` | PCA reduce states to D dimensions |
| `--ci` | Compute 95% confidence intervals |
| `-r FILE` | Save JSON report |
| `-q` | Quiet mode |
| `--explain` | Show detailed explanation |

## Comparison with score_lerobot_episodes

| Tool | Detects |
|------|---------|
| `score_lerobot_episodes` | Visual issues (blur, jitter) |
| `democlean` | Motion issues (jerky, inconsistent) |

They complement each other.

## License

MIT
