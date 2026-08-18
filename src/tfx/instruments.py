"""Instrument basket and per-asset-class trading-calendar facts.

Daily bars are bucketed by *session date*. Different asset classes close their day at different
wall-clock times (FX has no official close; conventionally 17:00 New York. Equity indices close
in their exchange-local timezone; crypto never closes). At daily resolution we record each
instrument's session-close timezone as metadata and treat the bar's date as its session date;
we do NOT bake intraday close times into the per-bar cache (that would add non-point-in-time
state). The cross-asset timing approximation this implies is documented in the data-quality
report.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class AssetClass(StrEnum):
    FX = "fx"
    EQUITY_INDEX = "equity_index"
    EQUITY = "equity"
    COMMODITY = "commodity"
    CRYPTO = "crypto"


# Python's date.weekday(): Monday=0 .. Sunday=6. Days on which an asset class is normally closed.
WEEKLY_CLOSED_DAYS: dict[AssetClass, frozenset[int]] = {
    AssetClass.FX: frozenset({5, 6}),            # Sat, Sun
    AssetClass.EQUITY_INDEX: frozenset({5, 6}),
    AssetClass.EQUITY: frozenset({5, 6}),
    AssetClass.COMMODITY: frozenset({5, 6}),
    AssetClass.CRYPTO: frozenset(),              # trades 24/7
}


class Instrument(BaseModel):
    """A tradable instrument. Immutable so it can be cached/compared safely."""

    model_config = ConfigDict(frozen=True)

    symbol: str                 # canonical display symbol, e.g. "EUR/USD"
    name: str
    asset_class: AssetClass
    session_close_tz: str       # IANA tz of the session close (metadata for daily bars)
    quote_currency: str | None = None

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier, e.g. 'EUR/USD' -> 'EUR_USD', 'S&P 500' -> 'SP_500'."""
        out = []
        for ch in self.symbol:
            if ch.isalnum():
                out.append(ch)
            elif ch in {"/", " ", "-"}:
                out.append("_")
            # drop everything else (e.g. '&')
        slug = "".join(out)
        # collapse repeated underscores deterministically
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug.strip("_")

    @property
    def weekly_closed_days(self) -> frozenset[int]:
        return WEEKLY_CLOSED_DAYS[self.asset_class]


# The v1 basket — breadth across lowly-correlated markets is THE dominant driver of
# trend-following performance (BUILD.md north star), so v1 spans FX majors/crosses, equity
# indices across three regions, commodities across energy/metals, and crypto. Bonds/rates are
# DEFERRED: the current provider carries US treasury series only as YIELDS, not prices — trading
# a yield series would corrupt returns and costs; revisit with a futures-price provider.
# Final list confirmed against the live broker later.
BASKET: tuple[Instrument, ...] = (
    Instrument(symbol="EUR/USD", name="Euro / US Dollar",
               asset_class=AssetClass.FX, session_close_tz="America/New_York",
               quote_currency="USD"),
    Instrument(symbol="USD/JPY", name="US Dollar / Japanese Yen",
               asset_class=AssetClass.FX, session_close_tz="America/New_York",
               quote_currency="JPY"),
    Instrument(symbol="GBP/USD", name="British Pound / US Dollar",
               asset_class=AssetClass.FX, session_close_tz="America/New_York",
               quote_currency="USD"),
    Instrument(symbol="AUD/JPY", name="Australian Dollar / Japanese Yen",
               asset_class=AssetClass.FX, session_close_tz="America/New_York",
               quote_currency="JPY"),
    Instrument(symbol="USD/CAD", name="US Dollar / Canadian Dollar",
               asset_class=AssetClass.FX, session_close_tz="America/New_York",
               quote_currency="CAD"),
    Instrument(symbol="GBP/CHF", name="British Pound / Swiss Franc",
               asset_class=AssetClass.FX, session_close_tz="America/New_York",
               quote_currency="CHF"),
    Instrument(symbol="NZD/USD", name="New Zealand Dollar / US Dollar",
               asset_class=AssetClass.FX, session_close_tz="America/New_York",
               quote_currency="USD"),
    Instrument(symbol="S&P 500", name="S&P 500 Index",
               asset_class=AssetClass.EQUITY_INDEX, session_close_tz="America/New_York",
               quote_currency="USD"),
    Instrument(symbol="DAX", name="DAX 40 Index",
               asset_class=AssetClass.EQUITY_INDEX, session_close_tz="Europe/Berlin",
               quote_currency="EUR"),
    Instrument(symbol="Nikkei 225", name="Nikkei 225 Index",
               asset_class=AssetClass.EQUITY_INDEX, session_close_tz="Asia/Tokyo",
               quote_currency="JPY"),
    Instrument(symbol="FTSE 100", name="FTSE 100 Index",
               asset_class=AssetClass.EQUITY_INDEX, session_close_tz="Europe/London",
               quote_currency="GBP"),
    Instrument(symbol="Gold", name="Gold (XAU/USD)",
               asset_class=AssetClass.COMMODITY, session_close_tz="America/New_York",
               quote_currency="USD"),
    Instrument(symbol="Silver", name="Silver (XAG/USD)",
               asset_class=AssetClass.COMMODITY, session_close_tz="America/New_York",
               quote_currency="USD"),
    Instrument(symbol="Oil", name="WTI Crude Oil",
               asset_class=AssetClass.COMMODITY, session_close_tz="America/New_York",
               quote_currency="USD"),
    Instrument(symbol="Nat Gas", name="Henry Hub Natural Gas",
               asset_class=AssetClass.COMMODITY, session_close_tz="America/New_York",
               quote_currency="USD"),
    Instrument(symbol="BTC/USD", name="Bitcoin / US Dollar",
               asset_class=AssetClass.CRYPTO, session_close_tz="UTC",
               quote_currency="USD"),
    Instrument(symbol="ETH/USD", name="Ether / US Dollar",
               asset_class=AssetClass.CRYPTO, session_close_tz="UTC",
               quote_currency="USD"),
)

BASKET_BY_SYMBOL: dict[str, Instrument] = {i.symbol: i for i in BASKET}
BASKET_BY_SLUG: dict[str, Instrument] = {i.slug: i for i in BASKET}


def default_symbols() -> list[str]:
    """Canonical symbols of the default basket, in basket order."""
    return [i.symbol for i in BASKET]


def get_instrument(symbol: str) -> Instrument:
    """Look up an instrument by canonical symbol or slug. Raises KeyError if unknown."""
    if symbol in BASKET_BY_SYMBOL:
        return BASKET_BY_SYMBOL[symbol]
    if symbol in BASKET_BY_SLUG:
        return BASKET_BY_SLUG[symbol]
    raise KeyError(
        f"Unknown instrument {symbol!r}. Known: {', '.join(BASKET_BY_SYMBOL)}"
    )
