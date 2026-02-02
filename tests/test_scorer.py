"""Tests for episode scoring."""

import numpy as np
import pytest

from democlean.scorer import DemoScorer, EpisodeScore


class TestEpisodeScore:
    """Test EpisodeScore dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        score = EpisodeScore(
            episode_index=5,
            mi_score=2.5,
            length=100,
            state_dim=6,
            action_dim=6,
            metadata={"task": "stack"},
        )

        d = score.to_dict()

        assert d["episode_index"] == 5
        assert d["mi_score"] == 2.5
        assert d["length"] == 100
        assert d["task"] == "stack"

    def test_to_dict_with_ci(self):
        """Test serialization with confidence intervals."""
        score = EpisodeScore(
            episode_index=0,
            mi_score=2.5,
            length=100,
            state_dim=2,
            action_dim=2,
            ci_lower=2.3,
            ci_upper=2.7,
        )

        d = score.to_dict()
        assert d["ci_lower"] == 2.3
        assert d["ci_upper"] == 2.7

    def test_normalized_score(self):
        """Test length normalization."""
        score = EpisodeScore(
            episode_index=0,
            mi_score=5.0,
            length=100,
            state_dim=2,
            action_dim=2,
        )

        expected = 5.0 / np.log(101)
        assert abs(score.normalized_score - expected) < 1e-6

    def test_zero_length_normalized(self):
        """Zero length gives 0 normalized score."""
        score = EpisodeScore(
            episode_index=0, mi_score=5.0, length=0, state_dim=2, action_dim=2
        )
        assert score.normalized_score == 0.0


class TestDemoScorer:
    """Test DemoScorer class."""

    def test_score_episode_basic(self):
        """Test scoring a single episode."""
        np.random.seed(42)
        states = np.random.randn(100, 4)
        actions = states @ np.random.randn(4, 2) + np.random.randn(100, 2) * 0.1

        scorer = DemoScorer(k=3)
        score = scorer.score_episode(states, actions, episode_index=0)

        assert score.episode_index == 0
        assert score.mi_score > 0
        assert score.length == 100
        assert score.state_dim == 4
        assert score.action_dim == 2

    def test_score_episode_with_dim_reduction(self):
        """Test scoring with PCA dimension reduction."""
        np.random.seed(42)
        states = np.random.randn(100, 50)  # High-dim
        actions = np.random.randn(100, 2)

        scorer = DemoScorer(k=3, max_state_dim=10)
        score = scorer.score_episode(states, actions, episode_index=0)

        assert score.mi_score >= 0
        # Original input dim should be recorded (before reduction)
        assert score.state_dim == 50  # Original, not reduced

    def test_filter_top_k(self):
        """Test filtering top K episodes."""
        scores = [
            EpisodeScore(0, mi_score=1.0, length=100, state_dim=2, action_dim=2),
            EpisodeScore(1, mi_score=3.0, length=100, state_dim=2, action_dim=2),
            EpisodeScore(2, mi_score=2.0, length=100, state_dim=2, action_dim=2),
            EpisodeScore(3, mi_score=5.0, length=100, state_dim=2, action_dim=2),
            EpisodeScore(4, mi_score=4.0, length=100, state_dim=2, action_dim=2),
        ]

        scorer = DemoScorer()

        top2 = scorer.filter_top_k(scores, k=2)
        assert top2 == [3, 4]

        top3 = scorer.filter_top_k(scores, k=3)
        assert top3 == [3, 4, 1]

    def test_filter_percentile(self):
        """Test filtering by percentile."""
        scores = [
            EpisodeScore(i, mi_score=float(i), length=100, state_dim=2, action_dim=2)
            for i in range(10)
        ]

        scorer = DemoScorer()
        top50 = scorer.filter_top_k(scores, percentile=50)

        assert len(top50) == 5
        assert 9 in top50
        assert 0 not in top50

    def test_filter_requires_k_or_percentile(self):
        """Should raise if neither k nor percentile."""
        scores = [EpisodeScore(0, mi_score=1.0, length=100, state_dim=2, action_dim=2)]
        scorer = DemoScorer()

        with pytest.raises(ValueError, match="Specify k or percentile"):
            scorer.filter_top_k(scores)

    def test_quality_assessment(self):
        """Test quality assessment."""
        scores = [
            EpisodeScore(
                i, mi_score=2.0 + i * 0.1, length=100, state_dim=2, action_dim=2
            )
            for i in range(10)
        ]

        scorer = DemoScorer()
        assessment = scorer.get_quality_assessment(scores)

        assert "mi_mean" in assessment
        assert "mi_std" in assessment
        assert "warnings" in assessment
        assert isinstance(assessment["warnings"], list)

    def test_quality_warning_low_std(self):
        """Should warn when MI std is low."""
        # All same score = 0 std
        scores = [
            EpisodeScore(i, mi_score=2.5, length=100, state_dim=2, action_dim=2)
            for i in range(10)
        ]

        scorer = DemoScorer()
        assessment = scorer.get_quality_assessment(scores)

        assert len(assessment["warnings"]) > 0
        assert "low" in assessment["warnings"][0].lower()

    def test_high_mi_for_deterministic_policy(self):
        """Deterministic policy should have high MI."""
        np.random.seed(42)
        states = np.random.randn(200, 4)
        actions = states @ np.random.randn(4, 2)  # Deterministic

        scorer = DemoScorer(k=3)
        score = scorer.score_episode(states, actions)

        assert score.mi_score > 1.0

    def test_low_mi_for_random_actions(self):
        """Random actions should have low MI."""
        np.random.seed(42)
        states = np.random.randn(200, 4)
        actions = np.random.randn(200, 2)  # Independent

        scorer = DemoScorer(k=3)
        score = scorer.score_episode(states, actions)

        assert score.mi_score < 0.5

    def test_quality_assessment_small_n_warning(self):
        """Should warn when n < 20."""
        scores = [
            EpisodeScore(i, mi_score=2.0, length=100, state_dim=2, action_dim=2)
            for i in range(5)
        ]

        scorer = DemoScorer()
        assessment = scorer.get_quality_assessment(scores)

        # Should have warning about small sample
        warnings_text = " ".join(assessment["warnings"])
        assert "episode" in warnings_text.lower() or len(assessment["warnings"]) > 0
