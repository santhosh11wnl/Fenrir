"""Report rendering -- CSV and Markdown, from rows the store already returned.

Pure functions over plain dicts, with no database access of their own. That
keeps the "what does a report contain" decision testable without a Postgres
instance, and stops report formatting from quietly growing its own queries.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any

#: Characters that make a spreadsheet treat a cell as a formula rather than
#: text. See `_safe_cell`.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _safe_cell(value: Any) -> str:
    """Render one cell, defusing spreadsheet formula injection.

    This is not paranoia about our own columns: ``subject`` embeds the
    visitor id, which the widget mints client-side and an attacker therefore
    controls. A visitor id of ``=HYPERLINK("http://evil","click")`` would
    execute as a formula the moment an operator opened the export in Excel or
    Sheets -- a stored attack whose payload travels through a CSV download and
    detonates on the analyst's machine.
    """
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    text = str(value)
    # A leading apostrophe is the conventional "treat as text" marker that
    # Excel, LibreOffice and Sheets all honour.
    return f"'{text}" if text.startswith(_FORMULA_PREFIXES) else text


def _write_csv(header: list[str], rows: list[list[Any]]) -> str:
    buffer = io.StringIO()
    # QUOTE_MINIMAL plus the prefix guard above: quoting alone does not stop
    # formula evaluation, since a quoted cell is still parsed as a formula.
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows([[_safe_cell(cell) for cell in row] for row in rows])
    return buffer.getvalue()


def turns_csv(rows: list[dict[str, Any]]) -> str:
    """One row per turn -- the raw export behind "Download report"."""
    header = [
        "occurred_at",
        "project",
        "conversation_id",
        "subject",
        "first_seen",
        "model",
        "duration_ms",
        "ttft_ms",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "tool_calls",
        "failed",
        "error_kind",
        "roles",
    ]
    return _write_csv(header, [[row.get(key) for key in header] for row in rows])


def cohort_csv(cohorts: dict[str, Any]) -> str:
    """Retention grid as a wide table, one row per cohort.

    Both the count and the percentage go in each cell (``12 (48.0%)``) rather
    than the percentage alone. A cohort of three people showing "100%
    retained" is noise, and a reader scanning percentages has no way to know
    that unless the denominator travels with them.
    """
    period = cohorts.get("period", "week")
    count = int(cohorts.get("periods", 0))
    header = ["cohort", "size"] + [f"{period}_{i}" for i in range(count)]

    rows: list[list[Any]] = []
    for entry in cohorts.get("cohorts", []):
        counts = entry.get("counts", [])
        retention = entry.get("retention", [])
        cells: list[Any] = [entry.get("cohort"), entry.get("size", 0)]
        for index in range(count):
            n = counts[index] if index < len(counts) else 0
            pct = retention[index] if index < len(retention) else 0.0
            cells.append(f"{n} ({pct * 100:.1f}%)" if n else "")
        rows.append(cells)
    return _write_csv(header, rows)


def summary_markdown(
    *,
    project: str,
    since: datetime,
    until: datetime,
    summary: dict[str, Any],
    breakdown: list[dict[str, Any]],
    cohorts: dict[str, Any] | None = None,
) -> str:
    """A human-readable report, for pasting into a doc or an email.

    Markdown rather than PDF: it renders in every destination an operator is
    likely to paste it into, diffs cleanly week over week, and needs no
    rendering dependency in the API image.
    """
    turns = summary.get("turns", 0) or 0
    lines = [
        f"# {project} — assistant report",
        "",
        f"**Window:** {since:%Y-%m-%d} to {until:%Y-%m-%d}",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Turns | {turns:,} |",
        f"| Conversations | {summary.get('conversations', 0):,} |",
        f"| Unique visitors | {summary.get('visitors', 0):,} |",
        f"| Turns per conversation | {_ratio(turns, summary.get('conversations', 0))} |",
        f"| Error rate | {(summary.get('error_rate', 0.0) * 100):.2f}% |",
        f"| Latency p50 | {_ms(summary.get('p50_ms'))} |",
        f"| Latency p95 | {_ms(summary.get('p95_ms'))} |",
        f"| Time to first token p50 | {_ms(summary.get('p50_ttft_ms'))} |",
        f"| Time to first token p95 | {_ms(summary.get('p95_ttft_ms'))} |",
        f"| Tool calls | {summary.get('tool_calls', 0):,} |",
        f"| Input tokens | {summary.get('input_tokens', 0):,} |",
        f"| Output tokens | {summary.get('output_tokens', 0):,} |",
        f"| Cache-read tokens | {summary.get('cache_read_tokens', 0):,} |",
        "",
    ]

    if breakdown:
        lines += [
            "## By model",
            "",
            "| Model | Turns | Errors | p50 |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| {row.get('value', '—')} | {row.get('turns', 0):,} | "
            f"{row.get('errors', 0):,} | {_ms(row.get('p50_ms'))} |"
            for row in breakdown
        ]
        lines.append("")

    if cohorts and cohorts.get("cohorts"):
        period = cohorts.get("period", "week")
        width = min(int(cohorts.get("periods", 0)), 6)
        lines += [
            f"## Retention by {period} cohort",
            "",
            "| Cohort | Size | " + " | ".join(f"+{i}" for i in range(width)) + " |",
            "| --- | --- | " + " | ".join("---" for _ in range(width)) + " |",
        ]
        for entry in cohorts["cohorts"]:
            retention = entry.get("retention", [])
            cells = [
                f"{retention[i] * 100:.0f}%" if i < len(retention) else "—"
                for i in range(width)
            ]
            lines.append(
                f"| {entry.get('cohort')} | {entry.get('size', 0):,} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")

    return "\n".join(lines)


def _ms(value: Any) -> str:
    if value is None:
        return "—"
    return f"{int(value) / 1000:.2f}s" if int(value) >= 1000 else f"{int(value)}ms"


def _ratio(numerator: Any, denominator: Any) -> str:
    n, d = int(numerator or 0), int(denominator or 0)
    return f"{n / d:.1f}" if d else "—"


__all__ = ["cohort_csv", "summary_markdown", "turns_csv"]
