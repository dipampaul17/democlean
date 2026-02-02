"""Integration tests on real LeRobot datasets."""

import numpy as np
import pytest

from democlean.scorer import DemoScorer


class TestRealDatasets:
    """Test scorer on real LeRobot datasets."""

    @pytest.mark.slow
    def test_pusht_scoring(self):
        """Test scoring on pusht dataset."""
        scorer = DemoScorer(k=3)
        scores = scorer.score_dataset("lerobot/pusht", max_episodes=20)

        assert len(scores) == 20
        assert all(s.mi_score >= 0 for s in scores)

        mi_values = [s.mi_score for s in scores]

        # Scores should have variance (not all same)
        assert np.std(mi_values) > 0.01, "Scores have no variance"

        print(f"\nScore range: [{min(mi_values):.3f}, {max(mi_values):.3f}]")
        print(f"Mean: {np.mean(mi_values):.3f}, Std: {np.std(mi_values):.3f}")

        # Check that filtering works
        top_10 = scorer.filter_top_k(scores, k=10)
        assert len(top_10) == 10

    @pytest.mark.slow
    def test_aloha_scoring(self):
        """Test scoring on aloha simulation dataset."""
        scorer = DemoScorer(k=3)
        scores = scorer.score_dataset(
            "lerobot/aloha_sim_transfer_cube_human",
            max_episodes=10,
        )

        assert len(scores) == 10

        mi_values = [s.mi_score for s in scores]
        print(f"\nAloha score range: [{min(mi_values):.3f}, {max(mi_values):.3f}]")

    def test_score_distribution_makes_sense(self):
        """Verify that scores differentiate episode quality."""
        scorer = DemoScorer(k=3)
        scores = scorer.score_dataset("lerobot/pusht", max_episodes=30)

        mi_values = [s.mi_score for s in scores]

        # Should have reasonable spread
        score_range = max(mi_values) - min(mi_values)
        assert score_range > 0.1, f"Score range too small: {score_range}"

        # Top and bottom should be different
        sorted_scores = sorted(scores, key=lambda s: s.mi_score)
        bottom_5_mean = np.mean([s.mi_score for s in sorted_scores[:5]])
        top_5_mean = np.mean([s.mi_score for s in sorted_scores[-5:]])

        assert top_5_mean > bottom_5_mean, "Top scores should be higher than bottom"
        print(f"\nBottom 5 mean: {bottom_5_mean:.3f}")
        print(f"Top 5 mean: {top_5_mean:.3f}")
        print(f"Difference: {top_5_mean - bottom_5_mean:.3f}")
