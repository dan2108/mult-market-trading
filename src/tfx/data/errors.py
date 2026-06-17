"""Exceptions for the data layer (kept small and explicit)."""

from __future__ import annotations


class DataError(Exception):
    """Base class for all data-layer errors."""


class DataQualityError(DataError):
    """Raised when raw data cannot be safely cleaned (e.g. non-positive prices, no usable price)."""


class CacheError(DataError):
    """Raised on cache I/O problems."""


class SchemaMismatchError(CacheError):
    """Raised when an on-disk cache was written by an incompatible schema/params and must not be
    silently trusted."""
