"""Tests for embedding encoders."""

import numpy as np
import pytest

from democlean.embeddings import Encoder, RawEmbedding, get_encoder


class TestRawEmbedding:
    """Test RawEmbedding passthrough."""

    def test_name(self):
        enc = RawEmbedding()
        assert enc.name == "raw"

    def test_encode_states_passthrough(self):
        enc = RawEmbedding()
        states = np.random.randn(100, 6).astype(np.float32)
        encoded = enc.encode_states(states)
        np.testing.assert_array_equal(states, encoded)

    def test_encode_actions_passthrough(self):
        enc = RawEmbedding()
        actions = np.random.randn(100, 2).astype(np.float32)
        encoded = enc.encode_actions(actions)
        np.testing.assert_array_equal(actions, encoded)

    def test_encode_episode(self):
        enc = RawEmbedding()
        states = np.random.randn(100, 6).astype(np.float32)
        actions = np.random.randn(100, 2).astype(np.float32)
        enc_s, enc_a = enc.encode_episode(states, actions)
        np.testing.assert_array_equal(states, enc_s)
        np.testing.assert_array_equal(actions, enc_a)


class TestGetEncoder:
    """Test encoder factory."""

    def test_get_raw(self):
        enc = get_encoder("raw")
        assert isinstance(enc, RawEmbedding)

    def test_invalid_encoder_raises(self):
        with pytest.raises(ValueError, match="Unknown encoder"):
            get_encoder("invalid")

    def test_get_hpt_requires_torch(self):
        """HPT should raise helpful error if torch not installed."""
        try:
            enc = get_encoder("hpt")
            # If it succeeds, torch is installed
            assert enc.name.startswith("hpt")
        except ImportError as e:
            # Should mention torch or democlean[hpt] installation
            err_msg = str(e).lower()
            assert "torch" in err_msg or "hpt" in err_msg


class TestEncoderInterface:
    """Test Encoder base class interface."""

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            Encoder()

    def test_raw_implements_interface(self):
        enc = RawEmbedding()
        assert hasattr(enc, "name")
        assert hasattr(enc, "encode_states")
        assert hasattr(enc, "encode_actions")
        assert hasattr(enc, "encode_episode")


# Mark HPT tests as slow and optional
@pytest.mark.slow
class TestHPTEmbedding:
    """Integration tests for HPT embedding (requires torch)."""

    @pytest.fixture
    def hpt_encoder(self):
        """Get HPT encoder, skip if torch not available."""
        try:
            from democlean.embeddings.hpt import HPTEmbedding
            return HPTEmbedding(model_size="small", device="cpu")
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_name(self, hpt_encoder):
        assert hpt_encoder.name == "hpt-small"

    def test_output_dim(self, hpt_encoder):
        assert hpt_encoder.output_dim == 128  # small model embed_dim

    def test_encode_states_shape(self, hpt_encoder):
        states = np.random.randn(50, 7).astype(np.float32)
        encoded = hpt_encoder.encode_states(states)
        assert encoded.shape == (50, hpt_encoder.output_dim)

    def test_encode_actions_shape(self, hpt_encoder):
        actions = np.random.randn(50, 6).astype(np.float32)
        encoded = hpt_encoder.encode_actions(actions)
        assert encoded.shape == (50, hpt_encoder.output_dim)

    def test_encode_episode(self, hpt_encoder):
        states = np.random.randn(50, 7).astype(np.float32)
        actions = np.random.randn(50, 6).astype(np.float32)
        enc_s, enc_a = hpt_encoder.encode_episode(states, actions)
        assert enc_s.shape == (50, hpt_encoder.output_dim)
        assert enc_a.shape == (50, hpt_encoder.output_dim)

    def test_different_input_dims(self, hpt_encoder):
        """Encoder should handle varying input dimensions."""
        states_7 = np.random.randn(20, 7).astype(np.float32)
        states_14 = np.random.randn(20, 14).astype(np.float32)

        enc_7 = hpt_encoder.encode_states(states_7)
        enc_14 = hpt_encoder.encode_states(states_14)

        # Both should have same output dim
        assert enc_7.shape == enc_14.shape

    def test_batching(self, hpt_encoder):
        """Test that large inputs are batched correctly."""
        from democlean.embeddings.hpt import HPTEmbedding

        enc = HPTEmbedding(model_size="small", device="cpu", batch_size=32)
        states = np.random.randn(100, 7).astype(np.float32)  # 100 > batch_size of 32
        encoded = enc.encode_states(states)
        assert encoded.shape == (100, enc.output_dim)

    def test_model_sizes(self, hpt_encoder):
        """Test different model sizes."""
        from democlean.embeddings.hpt import HPTEmbedding

        states = np.random.randn(10, 7).astype(np.float32)

        for size, expected_dim in [("small", 128), ("base", 256), ("large", 512)]:
            enc = HPTEmbedding(model_size=size, device="cpu")
            assert enc.output_dim == expected_dim

            encoded = enc.encode_states(states)
            assert encoded.shape == (10, expected_dim)


class TestScorerWithEncoder:
    """Test DemoScorer with different encoders."""

    def test_scorer_with_raw_encoder(self):
        from democlean import DemoScorer

        scorer = DemoScorer(k=3, encoder=RawEmbedding())
        states = np.random.randn(50, 6).astype(np.float32)
        actions = np.random.randn(50, 2).astype(np.float32)

        score = scorer.score_episode(states, actions)
        # MI can be 0 or negative for random/uncorrelated data
        assert isinstance(score.mi_score, float)
        assert score.metadata.get("encoder") == "raw"

    def test_scorer_default_encoder(self):
        from democlean import DemoScorer

        scorer = DemoScorer(k=3)
        assert scorer.encoder.name == "raw"

    @pytest.mark.slow
    def test_scorer_with_hpt_encoder(self):
        try:
            from democlean import DemoScorer, get_encoder
        except ImportError:
            pytest.skip("PyTorch not installed")

        try:
            encoder = get_encoder("hpt", model_size="small", device="cpu")
        except ImportError:
            pytest.skip("PyTorch not installed")

        scorer = DemoScorer(k=3, encoder=encoder)
        states = np.random.randn(50, 6).astype(np.float32)
        actions = np.random.randn(50, 2).astype(np.float32)

        score = scorer.score_episode(states, actions)
        # MI can be 0 or negative for random/uncorrelated data
        assert isinstance(score.mi_score, float)
        assert score.metadata.get("encoder") == "hpt-small"
