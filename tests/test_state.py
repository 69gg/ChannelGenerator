"""Tests for state persistence."""

from pathlib import Path

from channel_generator.analyzer import ChannelInfo
from channel_generator.state import State


def test_state_merge_and_preserve_first_seen(tmp_path: Path):
    """State should merge channels and preserve first_seen_at."""
    state_path = tmp_path / "state.json"
    state = State(state_path)

    ch = ChannelInfo(
        url="https://example.com/chat",
        name="Example",
        description="desc",
        models=["GPT-4o"],
        free_tier_desc="free",
        requires_auth=False,
        category="pure_chat",
        confidence="high",
        notes="",
        first_seen_at="2026-01-01T00:00:00+00:00",
        last_verified_at="2026-01-01T00:00:00+00:00",
    )
    state.merge([ch])
    state.save()

    state2 = State(state_path)
    ch2 = ChannelInfo(
        url="https://example.com/chat",
        name="Example Updated",
        description="desc",
        models=["GPT-4o"],
        free_tier_desc="free",
        requires_auth=False,
        category="pure_chat",
        confidence="high",
        notes="",
        first_seen_at="2026-02-01T00:00:00+00:00",
        last_verified_at="2026-02-01T00:00:00+00:00",
    )
    state2.merge([ch2])

    records = state2.list_channels()
    assert len(records) == 1
    assert records[0]["first_seen_at"] == "2026-01-01T00:00:00+00:00"
    assert records[0]["name"] == "Example Updated"


def test_state_list_sorted_and_limited(tmp_path: Path):
    """List should be sorted by last_verified_at desc and respect max_results."""
    state = State(tmp_path / "state.json")
    channels = [
        ChannelInfo(
            url=f"https://example{i}.com",
            name=f"Ex{i}",
            description="",
            models=[],
            free_tier_desc="",
            requires_auth=False,
            category="pure_chat",
            confidence="high",
            notes="",
            first_seen_at=f"2026-01-0{i}T00:00:00+00:00",
            last_verified_at=f"2026-01-0{i}T00:00:00+00:00",
        )
        for i in range(1, 4)
    ]
    state.merge(channels)
    records = state.list_channels(max_results=2)
    assert len(records) == 2
    assert records[0]["url"] == "https://example3.com"
