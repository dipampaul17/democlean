# democlean

[![PyPI](https://img.shields.io/pypi/v/democlean.svg)](https://pypi.org/project/democlean/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Quality scoring for robot demonstration datasets.

## The Problem

Robot learning datasets often contain bad demonstrations—jerky movements, hesitation, inconsistent timing. Training on these hurts policy performance. Manual review doesn't scale.

democlean automatically scores episodes by motion quality using mutual information (MI) between states and actions. Episodes with smooth, purposeful motion score high. Jerky, inconsistent episodes score low.

## Install

```bash
pip install democlean
```

## Quick Start

```bash
democlean analyze lerobot/pusht
```

Output:
```
Dataset lerobot/pusht
Episodes: 50 | Dims: 2→2

Distribution
  ████████████████████  High   30
  ██████████            Medium 15
  █████                 Low     5

Mean   2.55   (typical for human teleop)
Std    0.27

Flagged (lowest MI)
  ep  46  1.897
  ep   6  1.984
```

## What MI Measures

MI quantifies how predictable actions are given states.

**High MI** → actions are temporally smooth, low jerk, purposeful
**Low MI** → actions are jerky, hesitant, inconsistent timing

This is useful because motion quality correlates with demonstration quality. But MI is not a direct measure of task success—it measures *how* the robot moved, not *what* it achieved.

### Score Ranges

| MI | Interpretation |
|----|----------------|
| >3.0 | Very smooth |
| 2.0–3.0 | Typical human teleop |
| 1.0–2.0 | Moderate |
| <1.0 | Noisy/random |

## Filtering Episodes

Keep top 80%:
```bash
democlean analyze lerobot/pusht --keep 0.8
```

Drop below threshold:
```bash
democlean analyze lerobot/pusht --min-mi 2.0
```

Save report:
```bash
democlean analyze lerobot/pusht --keep 0.8 -r report.json
```

## Limitations

1. **Length correlation**: MI correlates with episode length (r≈0.8). Longer episodes score higher. Use `--normalize-length` to adjust.

2. **Not task success**: MI measures motion smoothness, not whether the task was completed. Use task-specific metrics for that.

3. **Sample size**: Works best with 50+ episodes. Small datasets may not show meaningful variation.

## Python API

```python
from democlean import DemoScorer

scorer = DemoScorer(k=3)
scores = scorer.score_dataset("lerobot/pusht")

# Filter
keep = scorer.filter_top_k(scores, percentile=80)
print(f"Keep episodes: {keep}")
```

## CLI Reference

```
democlean analyze <dataset> [options]

Options:
  --keep R            Keep top R fraction (0-1)
  --top-k K           Keep top K episodes
  --min-mi T          Drop episodes below MI threshold
  --normalize-length  Adjust for episode length
  -k N                KSG neighbors (default: 3)
  --max-dim D         PCA reduce to D dimensions
  --ci                Bootstrap confidence intervals
  -r FILE             Save JSON report
  -q                  Quiet mode (JSON output only)
  --explain           Show interpretation guide
```

## Credits

Built on the KSG mutual information estimator ([Kraskov et al., 2004](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138)).

Complements [score_lerobot_episodes](https://github.com/huggingface/lerobot/tree/main/lerobot/scripts) which catches visual issues—democlean catches motion issues.

## License

MIT
