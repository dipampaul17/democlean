#!/usr/bin/env python3
"""Scientific validation comparing raw vs HPT embeddings.

This script:
1. Loads a sample dataset (lerobot/pusht)
2. Computes MI scores with both raw and HPT encoders
3. Compares distributions and correlation with quality metrics
4. Reports statistical significance of differences

Usage:
    python scripts/validate_hpt.py [--max-episodes N]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from democlean import DemoScorer, get_encoder


def compute_jerk(actions: np.ndarray) -> float:
    """Compute mean jerk (3rd derivative) of action sequence.

    Lower jerk = smoother motion.
    """
    if len(actions) < 4:
        return 0.0

    velocity = np.diff(actions, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)

    return float(np.mean(np.abs(jerk)))


def compute_smoothness(actions: np.ndarray) -> float:
    """Compute smoothness via spectral arc length.

    Higher = smoother.
    """
    if len(actions) < 10:
        return 0.0

    velocity = np.diff(actions, axis=0)
    fft = np.fft.fft(velocity, axis=0)
    magnitude = np.abs(fft)

    arc_length = -np.sum(np.diff(magnitude, axis=0) ** 2)
    return float(arc_length / len(actions))


def compute_action_variance(actions: np.ndarray) -> float:
    """Compute variance of actions (higher = more varied motion)."""
    return float(np.var(actions))


def load_dataset_episodes(dataset_id: str, max_episodes: int | None = None):
    """Load episodes from a lerobot dataset."""
    from datasets import load_dataset

    print(f"Loading {dataset_id}...")
    ds = load_dataset(dataset_id, split="train")

    # Group by episode
    episodes = {}
    seen = set()

    for sample in ds:
        ep_idx = sample["episode_index"]

        if max_episodes and len(seen) >= max_episodes and ep_idx not in seen:
            break

        seen.add(ep_idx)

        if ep_idx not in episodes:
            episodes[ep_idx] = {"states": [], "actions": []}

        # Extract state (try common keys)
        state = None
        for key in ["observation.state", "observation.qpos", "state"]:
            if key in sample:
                state = sample[key]
                break

        if state is None:
            continue

        if isinstance(state, (list, tuple)):
            state = list(state)
        else:
            state = [state]

        action = sample["action"]
        if not isinstance(action, (list, tuple)):
            action = [action]

        episodes[ep_idx]["states"].append(state)
        episodes[ep_idx]["actions"].append(list(action))

    # Convert to arrays
    result = []
    for ep_idx, data in episodes.items():
        states = np.array(data["states"], dtype=np.float32)
        actions = np.array(data["actions"], dtype=np.float32)

        if len(states) < 10:
            continue

        result.append({
            "idx": ep_idx,
            "states": states,
            "actions": actions,
            "length": len(states),
            "jerk": compute_jerk(actions),
            "smoothness": compute_smoothness(actions),
            "action_variance": compute_action_variance(actions),
        })

    print(f"Loaded {len(result)} episodes")
    return result


def score_episodes(episodes: list, encoder_name: str, **encoder_kwargs) -> list[float]:
    """Score episodes with given encoder."""
    print(f"Scoring with encoder: {encoder_name}...")

    encoder = get_encoder(encoder_name, **encoder_kwargs)
    scorer = DemoScorer(k=3, encoder=encoder)

    scores = []
    for ep in episodes:
        score = scorer.score_episode(ep["states"], ep["actions"])
        scores.append(score.mi_score)

    return scores


def compare_correlations(
    mi_raw: np.ndarray,
    mi_hpt: np.ndarray,
    metric_values: np.ndarray,
    metric_name: str,
) -> dict:
    """Compare correlation of MI scores with a quality metric."""
    # Pearson correlations
    r_raw, p_raw = pearsonr(mi_raw, metric_values)
    r_hpt, p_hpt = pearsonr(mi_hpt, metric_values)

    # Spearman correlations (rank-based)
    rho_raw, _ = spearmanr(mi_raw, metric_values)
    rho_hpt, _ = spearmanr(mi_hpt, metric_values)

    # Fisher's z-test for comparing correlations
    n = len(mi_raw)
    if abs(r_raw) < 1 and abs(r_hpt) < 1:
        z_raw = np.arctanh(r_raw)
        z_hpt = np.arctanh(r_hpt)
        z_diff = (z_hpt - z_raw) / np.sqrt(2 / (n - 3))
        from scipy import stats
        p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
    else:
        z_diff = 0.0
        p_diff = 1.0

    return {
        "metric": metric_name,
        "raw": {
            "pearson_r": float(r_raw),
            "pearson_p": float(p_raw),
            "spearman_rho": float(rho_raw),
        },
        "hpt": {
            "pearson_r": float(r_hpt),
            "pearson_p": float(p_hpt),
            "spearman_rho": float(rho_hpt),
        },
        "comparison": {
            "r_diff": float(r_hpt - r_raw),
            "fisher_z": float(z_diff),
            "fisher_p": float(p_diff),
            "hpt_better": bool(abs(r_hpt) > abs(r_raw)),
            "significant": bool(p_diff < 0.05),
        },
    }


def run_validation(dataset_id: str = "lerobot/pusht", max_episodes: int | None = 50):
    """Run full validation comparing raw vs HPT embeddings."""
    print("=" * 70)
    print("HPT vs RAW EMBEDDING VALIDATION")
    print("=" * 70)

    # Load data
    episodes = load_dataset_episodes(dataset_id, max_episodes=max_episodes)

    if len(episodes) < 10:
        print("Error: Not enough episodes loaded")
        return None

    # Score with both encoders
    mi_raw = np.array(score_episodes(episodes, "raw"))
    mi_hpt = np.array(score_episodes(episodes, "hpt", model_size="base", device="cpu"))

    # Extract quality metrics
    jerk = np.array([ep["jerk"] for ep in episodes])
    smoothness = np.array([ep["smoothness"] for ep in episodes])
    action_var = np.array([ep["action_variance"] for ep in episodes])
    length = np.array([ep["length"] for ep in episodes])

    # Build results
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_id,
        "n_episodes": len(episodes),
        "mi_raw_stats": {
            "mean": float(np.mean(mi_raw)),
            "std": float(np.std(mi_raw)),
            "min": float(np.min(mi_raw)),
            "max": float(np.max(mi_raw)),
        },
        "mi_hpt_stats": {
            "mean": float(np.mean(mi_hpt)),
            "std": float(np.std(mi_hpt)),
            "min": float(np.min(mi_hpt)),
            "max": float(np.max(mi_hpt)),
        },
        "comparisons": [],
    }

    # Correlation between raw and HPT scores
    raw_hpt_corr, _ = spearmanr(mi_raw, mi_hpt)
    results["raw_hpt_spearman"] = float(raw_hpt_corr)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nRaw MI: mean={np.mean(mi_raw):.3f}, std={np.std(mi_raw):.3f}")
    print(f"HPT MI: mean={np.mean(mi_hpt):.3f}, std={np.std(mi_hpt):.3f}")
    print(f"Raw-HPT correlation (Spearman): {raw_hpt_corr:.3f}")

    # Compare correlations with quality metrics
    metrics = [
        ("jerk", -jerk, "Higher MI should mean lower jerk"),
        ("smoothness", smoothness, "Higher MI should mean higher smoothness"),
        ("action_variance", action_var, "Higher MI should mean more varied actions"),
        ("length", length, "Checking length correlation"),
    ]

    for name, values, description in metrics:
        comparison = compare_correlations(mi_raw, mi_hpt, values, name)
        results["comparisons"].append(comparison)

        print(f"\n{name.upper()}: {description}")
        print(f"  Raw:  r={comparison['raw']['pearson_r']:.3f} (p={comparison['raw']['pearson_p']:.4f})")
        print(f"  HPT:  r={comparison['hpt']['pearson_r']:.3f} (p={comparison['hpt']['pearson_p']:.4f})")
        if comparison["comparison"]["significant"]:
            winner = "HPT" if comparison["comparison"]["hpt_better"] else "Raw"
            print(f"  *** {winner} significantly better (p={comparison['comparison']['fisher_p']:.4f}) ***")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    hpt_wins = sum(1 for c in results["comparisons"] if c["comparison"]["hpt_better"])
    sig_wins = sum(
        1
        for c in results["comparisons"]
        if c["comparison"]["hpt_better"] and c["comparison"]["significant"]
    )

    print(f"HPT better on {hpt_wins}/{len(metrics)} metrics")
    print(f"HPT significantly better on {sig_wins}/{len(metrics)} metrics")

    if sig_wins >= 2:
        recommendation = "USE_HPT"
    elif sig_wins == 0 and hpt_wins <= 1:
        recommendation = "USE_RAW"
    else:
        recommendation = "INCONCLUSIVE"

    results["summary"] = {
        "hpt_better_count": hpt_wins,
        "hpt_significant_count": sig_wins,
        "total_metrics": len(metrics),
        "recommendation": recommendation,
    }

    print(f"\nRECOMMENDATION: {recommendation}")

    # Save results
    output_path = Path(__file__).parent.parent / "validation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate HPT vs Raw embeddings")
    parser.add_argument(
        "--dataset",
        default="lerobot/pusht",
        help="Dataset to use for validation",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=50,
        help="Maximum episodes to analyze",
    )
    args = parser.parse_args()

    run_validation(dataset_id=args.dataset, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
