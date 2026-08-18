"""Rendering: deterministic markdown (and dict) views of a Report.

Numbers are the artifact. Prose interpretation happens outside this module, in the research
workflow, clearly labelled observational.
"""

from __future__ import annotations

from .metrics import Report

_PERCENT = {
    "total_return", "cagr", "max_drawdown", "time_in_drawdown", "hit_rate",
    "avg_win", "avg_loss", "total_costs",
}
_TWO_DP = {"sharpe", "sortino", "annual_turnover", "mean_gross_exposure", "max_gross_exposure"}


def _format(key: str, value: object) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        if key in _PERCENT:
            signed = value < 0 or key in {"total_return", "cagr"}
            return f"{value:+.2%}" if signed else f"{value:.2%}"
        if key in _TWO_DP:
            return f"{value:.2f}"
        return f"{value:.6g}"
    return str(value)


def to_markdown(report: Report, title: str = "Backtest report") -> str:
    """One self-contained markdown document. Deterministic for a given Report."""
    lines = [f"# {title}", "", "## Metrics", "", "| metric | value |", "|---|---|"]
    for key, value in report.metrics.items():
        lines.append(f"| {key} | {_format(key, value)} |")

    lines += [
        "", "## Per-instrument contribution (net of costs)", "",
        "| instrument | contribution |", "|---|---|",
    ]
    for symbol, value in report.contributions.items():
        lines.append(f"| {symbol} | {value:+.4%} |")

    if not report.correlation.empty:
        symbols = list(report.correlation.columns)
        lines += ["", "## Attribution correlation", "", "| |" + "|".join(symbols) + "|"]
        lines.append("|---|" + "|".join("---" for _ in symbols) + "|")
        for row_symbol in symbols:
            cells = "|".join(f"{report.correlation.loc[row_symbol, c]:+.2f}" for c in symbols)
            lines.append(f"| {row_symbol} |" + cells + "|")

    lines += [
        "",
        "_Numbers deterministic; params provisional; costs pessimistic. The verdict on any edge",
        "comes from `tfx validate`, never from this report. Any prose interpretation is",
        "observational only and never feeds back into a trading decision._",
        "",
    ]
    return "\n".join(lines)
