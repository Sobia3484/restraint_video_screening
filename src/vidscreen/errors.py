class ExtractionError(Exception):
    """A recoverable extraction problem."""


class FatalExtractionError(ExtractionError):
    """Stop the whole run (e.g. invalid API key)."""


class CreditLimitReached(ExtractionError):
    """API credit budget exhausted (per-run limit, reserve, or HTTP 402)."""
