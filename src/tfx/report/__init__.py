"""Reporting (BUILD.md Step 8): the full metric set from a backtest run, presented honestly.

STRICTLY OBSERVATIONAL. This package computes and renders numbers; it imports nothing from
`tfx.core` and exposes no path that could mutate positions, signals, or params ("no LLM in the
live path" extends to: no REPORT in the live path). Any LLM interpretation happens OUTSIDE this
module, in the research workflow, on top of the rendered artifact -- clearly labelled
observational and never fed back into a decision. Numeric output is deterministic.
"""

from .metrics import Report, generate
from .render import to_markdown

__all__ = ["Report", "generate", "to_markdown"]
