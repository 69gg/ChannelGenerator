"""Tests for report generation."""

from pathlib import Path

from channel_generator.reporter import generate_report, write_report


def test_generate_report_format():
    """Report should contain expected sections and be sorted."""
    channels = [
        {
            "url": "https://a.com",
            "name": "A Chat",
            "description": "desc",
            "models": ["GPT-4o"],
            "free_tier_desc": "10/day",
            "requires_auth": False,
            "category": "pure_chat",
            "confidence": "high",
            "notes": "",
            "first_seen_at": "2026-07-04T00:00:00+00:00",
            "last_verified_at": "2026-07-05T00:00:00+00:00",
        },
        {
            "url": "https://b.com",
            "name": "B Tool",
            "description": "desc",
            "models": [],
            "free_tier_desc": "unlimited",
            "requires_auth": True,
            "category": "design_tool",
            "confidence": "medium",
            "notes": "note",
            "first_seen_at": "2026-07-03T00:00:00+00:00",
            "last_verified_at": "2026-07-04T00:00:00+00:00",
        },
    ]
    summary = {
        "summary": "Found some channels.",
        "highlights": ["A Chat is notable"],
        "category_counts": {"pure_chat": 1, "design_tool": 1},
        "confidence_counts": {"high": 1, "medium": 1, "low": 0},
    }
    report = generate_report(
        keywords=["free ai chat"],
        channels=channels,
        summary=summary,
        max_results=200,
    )

    assert "# 免费 LLM 渠道发现报告" in report
    assert "A Chat" in report
    assert "B Tool" in report
    assert "纯聊天站" in report
    assert "设计工具" in report
    assert "Found some channels." in report


def test_write_report(tmp_path: Path):
    """write_report should create the file."""
    path = tmp_path / "report.md"
    write_report("# Hello", path)
    assert path.read_text(encoding="utf-8") == "# Hello"
