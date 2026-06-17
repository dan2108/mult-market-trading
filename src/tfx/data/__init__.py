"""Data layer: acquisition, cleaning, alignment, adjustment, caching, and quality reporting.

The public surface that `backtest` consumes is :func:`tfx.data.loader.load`. Everything here is
deterministic and point-in-time correct (see :mod:`tfx.data.schema`).
"""
