"""Episode scoring using mutual information."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from democlean.mi import estimate_mi, estimate_mi_with_ci, reduce_dimensions


@dataclass
class EpisodeScore:
    """Score and metadata for a single episode."""

    episode_index: int
    mi_score: float
    length: int
    state_dim: int
    action_dim: int
    ci_lower: float | None = None
    ci_upper: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def normalized_score(self) -> float:
        """Score normalized by episode length."""
        if self.length == 0:
            return 0.0
        return self.mi_score / np.log(self.length + 1)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = {
            "episode_index": self.episode_index,
            "mi_score": round(self.mi_score, 4),
            "length": self.length,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        if self.ci_lower is not None:
            d["ci_lower"] = round(self.ci_lower, 4)
            d["ci_upper"] = round(self.ci_upper, 4)
        d.update(self.metadata)
        return d


class DemoScorer:
    """Score demonstrations by state-action MI.

    Example:
        scorer = DemoScorer(k=3)
        scores = scorer.score_dataset("lerobot/pusht")
        keep = scorer.filter_top_k(scores, percentile=80)
    """

    def __init__(
        self,
        k: int = 3,
        temporal_window: int = 1,
        state_keys: list[str] | None = None,
        action_key: str = "action",
        max_state_dim: int | None = None,
        bootstrap_ci: bool = False,
    ):
        """Initialize scorer.

        Args:
            k: KSG nearest neighbors
            temporal_window: State history frames
            state_keys: Dataset keys for state (auto-detect if None)
            action_key: Action key in dataset
            max_state_dim: Reduce state dim with PCA if exceeded
            bootstrap_ci: Compute 95% confidence intervals
        """
        self.k = k
        self.temporal_window = temporal_window
        self.state_keys = state_keys
        self.action_key = action_key
        self.max_state_dim = max_state_dim
        self.bootstrap_ci = bootstrap_ci

    def score_episode(
        self,
        states: NDArray[np.floating],
        actions: NDArray[np.floating],
        episode_index: int = 0,
        metadata: dict | None = None,
    ) -> EpisodeScore:
        """Score a single episode."""
        # Reduce dimensions if needed
        if self.max_state_dim and states.shape[1] > self.max_state_dim:
            states = reduce_dimensions(states, self.max_state_dim)

        if self.bootstrap_ci:
            mi, ci_lo, ci_hi = estimate_mi_with_ci(
                states, actions, k=self.k, n_bootstrap=50
            )
        else:
            mi = estimate_mi(
                states, actions, k=self.k, temporal_window=self.temporal_window
            )
            ci_lo, ci_hi = None, None

        return EpisodeScore(
            episode_index=episode_index,
            mi_score=mi,
            length=len(states),
            state_dim=states.shape[1] if states.ndim > 1 else 1,
            action_dim=actions.shape[1] if actions.ndim > 1 else 1,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            metadata=metadata or {},
        )

    def score_dataset(
        self,
        dataset,
        show_progress: bool = True,
        max_episodes: int | None = None,
    ) -> list[EpisodeScore]:
        """Score all episodes in a dataset.

        Args:
            dataset: HuggingFace dataset or repo ID string
            show_progress: Show progress bar
            max_episodes: Limit episodes (for testing)

        Returns:
            List of EpisodeScore
        """
        if isinstance(dataset, str):
            dataset = self._load_dataset(dataset)

        scores = []
        episodes = self._iter_episodes(dataset, max_episodes=max_episodes)

        if show_progress:
            from tqdm import tqdm

            episodes = tqdm(list(episodes), desc="Scoring episodes")

        for ep_idx, states, actions in episodes:
            score = self.score_episode(states, actions, episode_index=ep_idx)
            scores.append(score)

        return scores

    def filter_top_k(
        self,
        scores: list[EpisodeScore],
        k: int | None = None,
        percentile: float | None = None,
    ) -> list[int]:
        """Get indices of top-scoring episodes.

        Args:
            scores: From score_dataset
            k: Keep top k
            percentile: Keep top percentile (0-100)

        Returns:
            Episode indices to keep
        """
        if k is None and percentile is None:
            raise ValueError("Specify k or percentile")

        ranked = sorted(scores, key=lambda s: s.mi_score, reverse=True)

        if k is not None:
            keep = ranked[:k]
        else:
            n_keep = max(1, int(len(scores) * percentile / 100))
            keep = ranked[:n_keep]

        return [s.episode_index for s in keep]

    def get_quality_assessment(self, scores: list[EpisodeScore]) -> dict:
        """Assess overall dataset quality.

        Returns dict with stats and warnings.
        """
        mi = [s.mi_score for s in scores]
        std = np.std(mi)
        mean = np.mean(mi)

        assessment = {
            "n_episodes": len(scores),
            "mi_mean": round(mean, 3),
            "mi_std": round(std, 3),
            "mi_min": round(min(mi), 3),
            "mi_max": round(max(mi), 3),
            "warnings": [],
        }

        # Quality warnings
        n = len(scores)
        if std < 0.1:
            assessment["warnings"].append(
                f"MI std={std:.2f} is very low. Data may already be uniform quality. "
                "democlean works best when MI std > 0.2"
            )
        elif std < 0.2:
            assessment["warnings"].append(
                f"MI std={std:.2f} is low. Limited variation detected. "
                "Try analyzing more episodes or check if data is from simulation."
            )

        if mean < 1.0:
            assessment["warnings"].append(
                f"MI mean={mean:.2f} is low. Actions may be nearly random. "
                "Check data collection process or consider discarding this dataset."
            )

        if n < 20:
            assessment["warnings"].append(
                f"Only {n} episodes analyzed. "
                "For reliable statistics, use at least 20-50 episodes."
            )

        return assessment

    def _load_dataset(self, path_or_id: str):
        """Load dataset from path or HuggingFace."""
        from pathlib import Path

        from datasets import load_dataset

        if Path(path_or_id).exists():
            return load_dataset("parquet", data_dir=path_or_id, split="train")
        return load_dataset(path_or_id, split="train")

    def _iter_episodes(
        self,
        dataset,
        max_episodes: int | None = None,
    ) -> Iterator[tuple[int, NDArray, NDArray]]:
        """Iterate over episodes."""
        state_keys = self.state_keys or self._detect_state_keys(dataset)
        episodes_data = self._group_by_episode(dataset, state_keys, max_episodes)

        for ep_idx, (states, actions) in episodes_data.items():
            if len(states) > self.k + 1:
                yield ep_idx, states, actions

    def _detect_state_keys(self, dataset) -> list[str]:
        """Auto-detect state keys."""
        candidates = [
            "observation.state",
            "observation.qpos",
            "observation.joint_positions",
            "state",
        ]

        if hasattr(dataset, "features"):
            available = set(dataset.features.keys())
        elif hasattr(dataset, "column_names"):
            available = set(dataset.column_names)
        else:
            sample = next(iter(dataset))
            available = set(sample.keys())

        for key in candidates:
            if key in available:
                return [key]

        obs_keys = [k for k in available if k.startswith("observation.")]
        state_keys = [k for k in obs_keys if "image" not in k.lower()]

        if state_keys:
            return state_keys

        raise ValueError(f"Could not detect state keys. Available: {available}")

    def _group_by_episode(
        self,
        dataset,
        state_keys: list[str],
        max_episodes: int | None = None,
    ) -> dict[int, tuple[NDArray, NDArray]]:
        """Group frames by episode."""
        episodes: dict[int, tuple[list, list]] = {}
        seen = set()

        for sample in dataset:
            ep_idx = sample["episode_index"]

            if max_episodes and len(seen) >= max_episodes and ep_idx not in seen:
                break

            seen.add(ep_idx)

            if ep_idx not in episodes:
                episodes[ep_idx] = ([], [])

            # Extract state
            state_vals = []
            for key in state_keys:
                val = sample[key]
                if isinstance(val, (list, tuple)):
                    state_vals.extend(val)
                else:
                    state_vals.append(val)

            # Extract action
            action_val = sample[self.action_key]
            if not isinstance(action_val, (list, tuple)):
                action_val = [action_val]

            episodes[ep_idx][0].append(state_vals)
            episodes[ep_idx][1].append(list(action_val))

        # Convert to arrays
        return {
            ep_idx: (
                np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
            )
            for ep_idx, (states, actions) in episodes.items()
        }
