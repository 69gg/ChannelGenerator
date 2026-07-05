"""Tests for the CLI."""

from typer.testing import CliRunner

from channel_generator.cli import app

runner = CliRunner()


def test_version():
    """CLI should print the version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_run_help():
    """Run command should show help."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--llm-api-key" in result.output
    assert "--llm-max-tokens" in result.output
    assert "--llm-reasoning-" in result.output
    assert "--llm-summary-max" in result.output
    assert "--llm-summary-reas" in result.output
    assert "--keyword-count" in result.output
