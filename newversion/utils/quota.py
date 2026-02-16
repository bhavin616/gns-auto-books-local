"""Shared quota/rate-limit error detection for Gemini and APIs."""


def is_quota_error(e: Exception) -> bool:
    """Return True if the exception is a quota/rate-limit error (e.g. Gemini 429 / ResourceExhausted)."""
    err_msg = (str(e) or "").lower()
    err_name = type(e).__name__
    cause = getattr(e, "__cause__", None)
    cause_msg = (str(cause) or "").lower() if cause else ""
    combined = f"{err_name} {err_msg} {cause_msg}"
    return (
        "429" in combined
        or "quota" in combined
        or "resource exhausted" in combined
        or "resource_exhausted" in combined
        or "rate limit" in combined
        or "rate_limit" in combined
    )
