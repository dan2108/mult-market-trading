"""Tests for the basket expansion + the strict data-quality gate (provider onboarding).

The strict gate is what stands between a freshly-pulled provider dataset and the validate gate:
short history, unexpected gaps, and extreme one-bar moves (unadjusted futures rolls / bad ticks)
must fail loudly, per instrument, with exit code 1 at the CLI.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta

import pytest
from typer.testing import CliRunner

from tfx.cli import app
from tfx.costs.params import INSTRUMENT_OVERRIDES
from tfx.data.pull import pull
from tfx.data.quality import build_report
from tfx.data.sources.stooq import STOOQ_SYMBOLS
from tfx.instruments import BASKET, AssetClass


# --- basket integrity -------------------------------------------------------------------------
def test_basket_spans_asset_classes_with_unique_slugs():
    classes = {c: sum(1 for i in BASKET if i.asset_class == c) for c in AssetClass}
    assert classes[AssetClass.FX] == 7
    assert classes[AssetClass.EQUITY_INDEX] == 4
    assert classes[AssetClass.COMMODITY] == 4
    assert classes[AssetClass.CRYPTO] == 2
    assert len(BASKET) == 17
    slugs = [i.slug for i in BASKET]
    assert len(set(slugs)) == len(slugs)


def test_every_basket_symbol_has_cost_override_and_stooq_mapping():
    symbols = {i.symbol for i in BASKET}
    assert symbols <= set(INSTRUMENT_OVERRIDES), "explicit pessimistic costs per instrument"
    assert symbols <= set(STOOQ_SYMBOLS), "explicit provider mapping per instrument"


def test_crypto_trades_weekends_others_do_not():
    for instrument in BASKET:
        if instrument.asset_class is AssetClass.CRYPTO:
            assert instrument.weekly_closed_days == frozenset()
        else:
            assert instrument.weekly_closed_days == frozenset({5, 6})


# --- strict gate on the healthy fixture cache --------------------------------------------------
def test_strict_passes_on_healthy_data_when_history_suffices(cache_dir, symbols, window):
    start, end = window
    report = build_report(cache_dir, symbols, start, end)
    assert report.strict_failures(min_years=1.5) == []


def test_strict_min_history_fails_but_crypto_exempt(cache_dir, symbols, window):
    start, end = window  # the fixtures cover ~2 years, far short of 10
    report = build_report(cache_dir, symbols, start, end)
    failures = report.strict_failures(min_years=10.0)
    assert failures
    failed_symbols = {f.split(":")[0] for f in failures}
    assert "BTC/USD" not in failed_symbols and "ETH/USD" not in failed_symbols
    assert "EUR/USD" in failed_symbols


# --- strict gate catches crafted defects --------------------------------------------------------
def _write_fixture_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "open", "high", "low", "close"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _weekdays(start: date, n: int, skip: set[date] | None = None) -> list[date]:
    skip = skip or set()
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5 and d not in skip:
            days.append(d)
        d += timedelta(days=1)
    return days


def _flat_rows(days: list[date], price: float = 1.10) -> list[dict]:
    p = f"{price:.5f}"
    return [{"date": d.isoformat(), "open": p, "high": p, "low": p, "close": p} for d in days]


def test_strict_flags_extreme_move_roll_artifact(tmp_path, window):
    sample = tmp_path / "sample"
    sample.mkdir()
    days = _weekdays(date(2024, 1, 1), 30)
    rows = _flat_rows(days)
    for row in rows[15:]:  # +45% overnight jump: the shape of an unadjusted roll / bad tick
        row.update(open="1.60000", high="1.60000", low="1.60000", close="1.60000")
    _write_fixture_csv(sample / "EUR_USD.csv", rows)

    cache = tmp_path / "cache"
    pull(["EUR/USD"], days[0], days[-1], source="fixture", sample_dir=sample, cache_dir=cache)
    report = build_report(cache, ["EUR/USD"], days[0], days[-1])

    quality = report.instruments[0]
    assert len(quality.extreme_moves) == 1
    day, move = quality.extreme_moves[0]
    assert day == days[15].isoformat()
    assert move == pytest.approx(0.5 / 1.1, rel=1e-9)
    failures = report.strict_failures(min_years=0.01)
    assert any("extreme one-bar move" in f for f in failures)

    # same data under a looser threshold passes the move screen
    loose = build_report(cache, ["EUR/USD"], days[0], days[-1], extreme_move_threshold=0.60)
    assert not any("extreme" in f for f in loose.strict_failures(min_years=0.01))


def test_strict_flags_unexpected_gap(tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    all_days = _weekdays(date(2024, 1, 1), 40)
    hole = set(all_days[15:25])  # 10 consecutive missing weekdays > max_weekday_gap_days
    kept = [d for d in all_days if d not in hole]
    _write_fixture_csv(sample / "EUR_USD.csv", _flat_rows(kept))

    cache = tmp_path / "cache"
    pull(
        ["EUR/USD"], all_days[0], all_days[-1],
        source="fixture", sample_dir=sample, cache_dir=cache,
    )
    report = build_report(cache, ["EUR/USD"], all_days[0], all_days[-1])

    assert report.instruments[0].n_unexpected_gaps == 1
    failures = report.strict_failures(min_years=0.01)
    assert any("unexpected gap" in f for f in failures)


# --- CLI exit codes -----------------------------------------------------------------------------
def test_cli_strict_gate_exit_codes(cache_dir, window):
    start, end = window
    runner = CliRunner()
    base = [
        "data", "report", "--cache-dir", str(cache_dir),
        "--start", str(start), "--end", str(end),
    ]

    failing = runner.invoke(app, [*base, "--strict"])  # default --min-years 10 vs ~2y fixtures
    assert failing.exit_code == 1

    passing = runner.invoke(app, [*base, "--strict", "--min-years", "1.5"])
    assert passing.exit_code == 0, passing.output
