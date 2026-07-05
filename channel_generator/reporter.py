"""Markdown report generation."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _category_label(category: str) -> str:
    labels = {
        "pure_chat": "纯聊天站",
        "design_tool": "设计工具",
        "coding_tool": "代码助手",
        "writing_tool": "写作/办公助手",
        "aggregator": "聚合平台",
        "roleplay": "角色扮演",
        "other": "其他",
    }
    return labels.get(category, category)


def generate_report(
    keywords: list[str],
    channels: list[dict[str, Any]],
    summary: dict[str, Any],
    max_results: int,
) -> str:
    """Generate a Markdown report from discovered channels.

    Args:
        keywords: Search keywords used.
        channels: Channel records sorted by recency.
        summary: Structured summary from LLM.
        max_results: Maximum results setting.

    Returns:
        Markdown report string.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    total = len(channels)
    conf = summary.get("confidence_counts", {"high": 0, "medium": 0, "low": 0})
    categories = summary.get("category_counts", {})

    lines: list[str] = [
        "# 免费 LLM 渠道发现报告",
        "",
        f"生成时间：{now}",
        f"本次搜索关键词（LLM 生成）：{', '.join(keywords[:30])}",
        f"报告数量上限：{max_results}",
        "",
        "## 概览",
        "",
        f"- 总数：{total}",
        f"- 高置信度：{conf.get('high', 0)}",
        f"- 中置信度：{conf.get('medium', 0)}",
        f"- 低置信度：{conf.get('low', 0)}",
        "",
        "### 执行摘要",
        "",
        summary.get("summary", ""),
        "",
    ]

    if summary.get("highlights"):
        lines.extend(["### 亮点", ""])
        for h in summary["highlights"]:
            lines.append(f"- {h}")
        lines.append("")

    if categories:
        lines.extend(["### 分类分布", ""])
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            lines.append(f"- {_category_label(cat)}：{count}")
        lines.append("")

    lines.extend(["## 渠道列表（按最近发现/验证时间倒序）", ""])

    for idx, ch in enumerate(channels, start=1):
        lines.extend(
            [
                f"### {idx}. {ch.get('name', 'Unknown')}",
                "",
                f"- URL: {ch.get('url', '')}",
                f"- 类别：{_category_label(ch.get('category', 'other'))}",
                f"- 模型：{', '.join(ch.get('models', [])) or '未明确'}",
                f"- 免费额度：{ch.get('free_tier_desc', '未明确')}",
                f"- 需要登录：{'是' if ch.get('requires_auth') else '否'}",
                f"- 置信度：{ch.get('confidence', 'low')}",
                f"- 首次发现：{ch.get('first_seen_at', '')[:10]}",
                f"- 最近验证：{ch.get('last_verified_at', '')[:10]}",
            ]
        )
        if ch.get("notes"):
            lines.append(f"- 备注：{ch.get('notes')}")
        lines.extend(["", "---", ""])

    return "\n".join(lines)


def write_report(report: str, path: Path) -> None:
    """Write report to disk.

    Args:
        report: Markdown report string.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
