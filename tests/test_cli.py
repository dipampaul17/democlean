"""Tests for CLI functionality."""

import json
import subprocess
import sys
from pathlib import Path


def run_cli(*args):
    """Run democlean CLI and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "democlean.cli"] + list(args),
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    return result.returncode, result.stdout, result.stderr


class TestCLIValidation:
    """Test input validation."""

    def test_k_zero_rejected(self):
        """k=0 should error."""
        code, stdout, stderr = run_cli("analyze", "lerobot/pusht", "-n", "2", "-k", "0")
        assert code != 0 or "error" in stdout.lower() or "error" in stderr.lower()

    def test_negative_keep_rejected(self):
        """Negative --keep should error."""
        code, stdout, stderr = run_cli(
            "analyze", "lerobot/pusht", "-n", "2", "--keep", "-0.5"
        )
        assert code != 0 or "error" in stdout.lower()

    def test_keep_over_one_rejected(self):
        """--keep > 1 should error."""
        code, stdout, stderr = run_cli(
            "analyze", "lerobot/pusht", "-n", "2", "--keep", "1.5"
        )
        assert code != 0 or "error" in stdout.lower()


class TestCLIFeatures:
    """Test CLI features work."""

    def test_min_mi_filter(self):
        """--min-mi should filter episodes."""
        code, stdout, stderr = run_cli(
            "analyze", "lerobot/pusht", "-n", "5", "--min-mi", "2.5", "-q"
        )
        assert code == 0

    def test_normalize_length(self):
        """--normalize-length should work."""
        code, stdout, stderr = run_cli(
            "analyze", "lerobot/pusht", "-n", "5", "--normalize-length"
        )
        assert code == 0

    def test_explain_flag(self):
        """--explain should show extra info."""
        code, stdout, stderr = run_cli(
            "analyze", "lerobot/pusht", "-n", "5", "--explain"
        )
        assert code == 0
        # Should contain explanation text
        assert "MI" in stdout or "mutual" in stdout.lower()


class TestCLIOutput:
    """Test CLI output format."""

    def test_json_report_structure(self):
        """JSON report should have required fields."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            report_path = f.name

        code, _, _ = run_cli(
            "analyze", "lerobot/pusht", "-n", "5", "-r", report_path, "-q"
        )
        assert code == 0

        report = json.loads(Path(report_path).read_text())
        assert "version" in report
        assert "dataset" in report
        assert "assessment" in report
        assert "scores" in report
        assert "mi_mean" in report["assessment"]

        Path(report_path).unlink()

    def test_quiet_mode_minimal_output(self):
        """Quiet mode should have minimal output."""
        code, stdout, _ = run_cli("analyze", "lerobot/pusht", "-n", "3", "-q")
        assert code == 0
        # Should not have progress bars or detailed stats
        lines = [line for line in stdout.strip().split("\n") if line.strip()]
        assert len(lines) <= 5  # Very minimal output
