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
