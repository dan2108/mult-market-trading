"""`tfx` — the command-line surface for the platform.

v1 exposes the data layer: `tfx data pull` (fetch -> clean -> align -> cache) and
`tfx data report` (data-quality report). Later phases add `backtest`, `validate`, `report`, and
the Phase-2 live commands behind the same entrypoint.
"""

from __future__ import annotations

import sys
from typing import NoReturn

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import get_settings
from .data.errors import DataError
from .data.pull import pull as pull_data
from .data.quality import TABLE_COLUMNS, build_report
from .data.sources import available_sources
from .instruments import BASKET, get_instrument

# Windows terminals often use a cp1252 code page; degrade any un-encodable glyph to '?' instead
# of crashing. Intentional CLI output is ASCII-only, so this only ever affects unexpected input.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, OSError):  # pragma: no cover - stream may not support reconfigure
        pass

# Defaults match the bundled deterministic fixtures (2023-01-02 .. 2024-12-31).
DEFAULT_START = "2023-01-02"
DEFAULT_END = "2024-12-31"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Diversified multi-market (FX/stock/crypto) trend-following trading platform.",
)
data_app = typer.Typer(no_args_is_help=True, help="Data acquisition, caching and quality.")
app.add_typer(data_app, name="data")
backtest_app = typer.Typer(no_args_is_help=True, help="Deterministic backtest of the shared core.")
app.add_typer(backtest_app, name="backtest")
validate_app = typer.Typer(no_args_is_help=True, help="THE GATE: walk-forward OOS validation.")
app.add_typer(validate_app, name="validate")

console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"tfx {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Diversified multi-market trend-following trading platform."""


def _parse_symbols(symbols: str | None) -> list[str]:
    if not symbols:
        return [i.symbol for i in BASKET]
    return [s.strip() for s in symbols.split(",") if s.strip()]


def _fail(message: str) -> NoReturn:
    err_console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """Show the active configuration and the instrument basket."""
    settings = get_settings()
    console.print(f"[bold]tfx[/bold] {__version__}")
    console.print(f"cache_dir : {settings.cache_dir}")
    console.print(f"sample_dir: {settings.sample_dir}")
    console.print(f"source    : {settings.data.provider}")
    table = Table(title="Instrument basket")
    for col in ("Symbol", "Name", "Class", "Quote", "Session close tz"):
        table.add_column(col)
    for instrument in BASKET:
        table.add_row(
            instrument.symbol, instrument.name, instrument.asset_class.value,
            instrument.quote_currency or "-", instrument.session_close_tz,
        )
    console.print(table)


@data_app.command("pull")
def data_pull(
    symbols: str | None = typer.Option(
        None, "--symbols", "-s", help="Comma-separated symbols (default: full basket)."
    ),
    start: str = typer.Option(DEFAULT_START, "--start", help="Start date YYYY-MM-DD."),
    end: str = typer.Option(DEFAULT_END, "--end", help="End date YYYY-MM-DD."),
    source: str | None = typer.Option(
        None, "--source", help=f"Data source. Available: {', '.join(available_sources())}."
    ),
    sample_dir: str | None = typer.Option(None, "--sample-dir", help="Override fixture dir."),
    cache_dir: str | None = typer.Option(None, "--cache-dir", help="Override cache dir."),
) -> None:
    """Fetch -> clean -> align -> cache the basket for a date range."""
    settings = get_settings()
    syms = _parse_symbols(symbols)
    src = source or settings.data.provider
    cdir = cache_dir or settings.cache_dir
    sdir = sample_dir or settings.sample_dir

    try:
        result = pull_data(syms, start, end, source=src, cache_dir=cdir, sample_dir=sdir)
    except (DataError, FileNotFoundError, NotImplementedError, ValueError, KeyError) as exc:
        _fail(str(exc))

    console.print(
        f"[green]Pulled[/green] {result.n_instruments} instrument(s) "
        f"-> {result.cache_dir}"
    )
    table = Table(title=f"Cached  {start} to {end}  (source: {src})")
    for col in ("Instrument", "Bars", "First", "Last"):
        table.add_column(col)
    instruments = result.manifest["instruments"]
    for sym in syms:
        entry = instruments[get_instrument(sym).slug]
        table.add_row(
            entry["symbol"], str(entry["rows"]),
            (entry["first"] or "-")[:10], (entry["last"] or "-")[:10],
        )
    console.print(table)
    console.print("Next: run [bold]tfx data report[/bold] to inspect coverage and gaps.")


@backtest_app.command("run")
def backtest_run(
    symbols: str | None = typer.Option(
        None, "--symbols", "-s", help="Comma-separated symbols (default: full basket)."
    ),
    start: str = typer.Option(DEFAULT_START, "--start", help="Start date YYYY-MM-DD."),
    end: str = typer.Option(DEFAULT_END, "--end", help="End date YYYY-MM-DD."),
    cache_dir: str | None = typer.Option(None, "--cache-dir", help="Override cache dir."),
) -> None:
    """Run the bar-by-bar backtest on the cached panel (default core params).

    Research output only -- the honest verdict on an edge comes from `tfx validate`, never from
    a single backtest run.
    """
    from .backtest import run as run_backtest
    from .data.loader import load

    settings = get_settings()
    syms = _parse_symbols(symbols)
    cdir = cache_dir or settings.cache_dir

    try:
        panel = load(syms, start, end, cache_dir=cdir)
        result = run_backtest(panel)
    except (DataError, FileNotFoundError, KeyError, ValueError) as exc:
        _fail(str(exc))

    equity = result.equity_curve
    drawdown = (equity / equity.cummax() - 1.0).min()
    total_costs = result.costs["amount"].sum() if len(result.costs) else 0.0
    table = Table(title=f"Backtest  {start} to {end}  ({len(syms)} instruments)")
    for col in ("Bars", "Trades", "Final equity", "Total return", "Max drawdown", "Costs paid"):
        table.add_column(col)
    table.add_row(
        str(len(equity)), str(len(result.trades)), f"{equity.iloc[-1]:.4f}",
        f"{equity.iloc[-1] - 1.0:+.2%}", f"{drawdown:.2%}", f"{total_costs:.4%}",
    )
    console.print(table)
    console.print(
        "[dim]Provisional params; costs pessimistic. The gate is [bold]tfx validate[/bold], "
        "not this table.[/dim]"
    )


@validate_app.command("run")
def validate_run(
    symbols: str | None = typer.Option(
        None, "--symbols", "-s", help="Comma-separated symbols (default: full basket)."
    ),
    start: str = typer.Option(DEFAULT_START, "--start", help="Start date YYYY-MM-DD."),
    end: str = typer.Option(DEFAULT_END, "--end", help="End date YYYY-MM-DD."),
    cache_dir: str | None = typer.Option(None, "--cache-dir", help="Override cache dir."),
    train_bars: int | None = typer.Option(
        None, "--train-bars",
        help="Override the pre-registered train window (bars). Marks the run NON-CANONICAL.",
    ),
    test_bars: int | None = typer.Option(
        None, "--test-bars",
        help="Override the pre-registered test window (bars). Marks the run NON-CANONICAL.",
    ),
    step_bars: int | None = typer.Option(
        None, "--step-bars",
        help="Override the pre-registered roll step (bars). Marks the run NON-CANONICAL.",
    ),
) -> None:
    """Run the walk-forward gate on the cached panel. Exit code 0 ONLY on a PASS verdict.

    Thresholds and the parameter grid are pre-registered in ValidateProtocol and echoed into
    the result -- change them BEFORE a run, never after seeing numbers. The fold-geometry flags
    exist for research/exploration; overriding any of them marks the run NON-CANONICAL in the
    output, loudly, so re-running with different geometry until one happens to PASS can't be
    mistaken for -- or quietly reported as -- the pre-registered gate result.
    """
    from pydantic import ValidationError

    from .data.loader import load
    from .validate import ValidateProtocol, Verdict
    from .validate import run as run_validate

    settings = get_settings()
    syms = _parse_symbols(symbols)
    cdir = cache_dir or settings.cache_dir

    try:
        default_protocol = ValidateProtocol()
        protocol = ValidateProtocol(
            train_bars=train_bars if train_bars is not None else default_protocol.train_bars,
            test_bars=test_bars if test_bars is not None else default_protocol.test_bars,
            step_bars=step_bars if step_bars is not None else default_protocol.step_bars,
        )
        is_canonical = protocol == default_protocol
        panel = load(syms, start, end, cache_dir=cdir)
        result = run_validate(panel, protocol)
    except (DataError, FileNotFoundError, KeyError, ValueError, ValidationError) as exc:
        _fail(str(exc))

    override_note = (
        "" if is_canonical else "  [yellow]<- OVERRIDDEN, not the pre-registered default[/yellow]"
    )
    console.print(
        f"Geometry: train={protocol.train_bars} test={protocol.test_bars} "
        f"step={protocol.step_bars} bars" + override_note
    )
    if not is_canonical:
        err_console.print(
            "[yellow]NON-CANONICAL RUN[/yellow]: fold geometry differs from "
            "ValidateProtocol()'s pre-registered default. This verdict is exploratory -- "
            "re-registering the gate's real geometry means changing ValidateProtocol's "
            "defaults, not passing CLI flags until one run passes."
        )

    table = Table(title=f"Walk-forward folds  ({len(result.folds)})")
    for col in ("Fold", "Test window", "Selected", "Train SR", "Test SR", "Trades"):
        table.add_column(col)
    for fold in result.folds:
        table.add_row(
            str(fold.fold),
            f"{fold.test_start.date()} .. {fold.test_end.date()}",
            fold.selected.label(),
            f"{fold.train_sharpe:.2f}", f"{fold.test_sharpe:.2f}", str(fold.test_trades),
        )
    console.print(table)

    metrics = result.oos_metrics
    console.print(
        f"OOS: {metrics['n_bars']} bars, {metrics['n_trades']} trades | "
        f"Sharpe raw {metrics['sharpe_raw']:.2f} -> haircut "
        f"[bold]{metrics['sharpe_haircut']:.2f}[/bold] "
        f"(min {result.protocol.min_sharpe_after_haircut:g}) | "
        f"t-stat {metrics['t_stat']:.2f} (min {result.protocol.min_tstat:g}) | "
        f"maxDD {metrics['max_drawdown']:.1%} | DSR {metrics['deflated_sharpe']:.3f}"
    )
    robustness = result.robustness
    console.print(
        f"Robustness: modal {robustness['modal_candidate']} in "
        f"{robustness['modal_share']:.0%} of folds; grid-mean OOS Sharpe "
        f"{robustness['grid_mean_oos_sharpe']:.2f}"
    )
    canonical_note = "" if is_canonical else " [yellow](NON-CANONICAL -- exploratory)[/yellow]"
    if result.verdict is Verdict.PASS:
        console.print(f"[green]VERDICT: PASS[/green]{canonical_note}")
    else:
        err_console.print(f"[red]VERDICT: {result.verdict.value.upper()}[/red]{canonical_note}")
        raise typer.Exit(code=1)


@app.command("report")
def report_command(
    symbols: str | None = typer.Option(
        None, "--symbols", "-s", help="Comma-separated symbols (default: full basket)."
    ),
    start: str = typer.Option(DEFAULT_START, "--start", help="Start date YYYY-MM-DD."),
    end: str = typer.Option(DEFAULT_END, "--end", help="End date YYYY-MM-DD."),
    cache_dir: str | None = typer.Option(None, "--cache-dir", help="Override cache dir."),
    out: str | None = typer.Option(
        None, "--out", help="Also write the full report as markdown to this path."
    ),
) -> None:
    """Full metric report of a backtest over the cached panel. Observational only."""
    from pathlib import Path

    from .backtest import run as run_backtest
    from .data.loader import load
    from .report import generate, to_markdown

    settings = get_settings()
    syms = _parse_symbols(symbols)
    cdir = cache_dir or settings.cache_dir

    try:
        panel = load(syms, start, end, cache_dir=cdir)
        report = generate(run_backtest(panel))
    except (DataError, FileNotFoundError, KeyError, ValueError) as exc:
        _fail(str(exc))

    table = Table(title=f"Backtest report  {start} to {end}")
    table.add_column("Metric")
    table.add_column("Value")
    from .report.render import _format

    for key, value in report.metrics.items():
        table.add_row(key, _format(key, value))
    console.print(table)

    contrib = Table(title="Per-instrument contribution (net of costs)")
    contrib.add_column("Instrument")
    contrib.add_column("Contribution")
    for symbol, value in report.contributions.items():
        contrib.add_row(str(symbol), f"{value:+.4%}")
    console.print(contrib)

    if out is not None:
        Path(out).write_text(to_markdown(report), encoding="utf-8")
        console.print(f"[green]Wrote[/green] {out}")
    console.print(
        "[dim]Observational only -- numbers deterministic, interpretation lives in the "
        "research workflow. The gate is [bold]tfx validate[/bold].[/dim]"
    )


@data_app.command("report")
def data_report(
    symbols: str | None = typer.Option(None, "--symbols", "-s", help="Comma-separated symbols."),
    start: str = typer.Option(DEFAULT_START, "--start", help="Start date YYYY-MM-DD."),
    end: str = typer.Option(DEFAULT_END, "--end", help="End date YYYY-MM-DD."),
    cache_dir: str | None = typer.Option(None, "--cache-dir", help="Override cache dir."),
    max_gap: int | None = typer.Option(
        None, "--max-gap", help="Max consecutive missing weekdays before a gap is 'unexpected'."
    ),
) -> None:
    """Print a data-quality report from the cache (coverage, gaps, repairs)."""
    settings = get_settings()
    syms = _parse_symbols(symbols)
    cdir = cache_dir or settings.cache_dir

    try:
        report = build_report(cdir, syms, start, end, max_weekday_gap_days=max_gap)
    except (DataError, FileNotFoundError, KeyError, ValueError) as exc:
        _fail(str(exc))

    table = Table(title=f"Data-quality report  {report.start} to {report.end}")
    for col in TABLE_COLUMNS:
        table.add_column(col)
    for quality in report.instruments:
        table.add_row(*quality.table_row())
    console.print(table)
    console.print(
        "[dim]Cov% = % of expected trading days; Gaps = holiday/unexpected; "
        "Imp/Rep = imputed/repaired bars.[/dim]"
    )

    if report.has_unexpected_gaps:
        console.print(
            "[yellow]Unexpected gaps (investigate before backtesting):[/yellow]"
        )
        for quality in report.instruments:
            for first, last, length in quality.unexpected_runs:
                console.print(f"  {quality.symbol}: {first} to {last} ({length} weekdays)")
    else:
        console.print("[green]No unexpected gaps.[/green]")

    for symbol, dups, dropped in report.cleaning_anomalies:
        console.print(
            f"[yellow]cleaning:[/yellow] {symbol}: {dups} duplicate(s) resolved, "
            f"{dropped} unusable bar(s) dropped"
        )

    console.print(
        "[dim]Note: daily bars use each instrument's native session date; cross-asset timing is "
        "approximate at daily resolution (see tfx.instruments).[/dim]"
    )


if __name__ == "__main__":
    app()
