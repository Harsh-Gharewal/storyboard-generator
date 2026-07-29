"""Token logger — tracks requested vs. cached token counts per Gemini call.

Appends each call's token breakdown as a JSON line to:
    /storage/logs/token-usage.log

Exposes get_summary(script_id) to calculate total vs cached token savings
and report real percentage savings.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

# ── In-memory running totals ─────────────────────────────────────────

_totals = {
    "total_requested": 0,
    "total_cached": 0,
    "call_count": 0,
}


def _get_log_file_path() -> Path:
    """Return the path to /storage/logs/token-usage.log, creating parent dirs."""
    log_dir = Path(settings.STORAGE_DIR) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "token-usage.log"


def log_token_usage(
    *,
    model: str,
    prompt_tokens: Optional[int] = None,
    requested_tokens: Optional[int] = None,
    cached_tokens: int = 0,
    script_id: Optional[str] = None,
    shot_id: Optional[str] = None,
    call_type: Optional[str] = None,
    note: Optional[str] = None,
) -> dict:
    """Log a single Gemini API call's token usage.

    Supports both prompt_tokens and requested_tokens parameter names for
    backwards compatibility. Appends entry to /storage/logs/token-usage.log.

    Args:
        model: Gemini model identifier (e.g. gemini-3.5-flash).
        prompt_tokens: Number of prompt tokens requested.
        requested_tokens: Alias for prompt_tokens.
        cached_tokens: Number of tokens served from cache.
        script_id: Optional script ID for per-script tracking.
        shot_id: Optional shot ID for shot-level traceability.
        call_type: Optional label for the call (e.g. "script_parse", "shot_gen").
        note: Optional free-text annotation.

    Returns:
        The written log entry dict.
    """
    total_prompt_tokens = prompt_tokens if prompt_tokens is not None else (requested_tokens or 0)
    cached = cached_tokens or 0
    fresh = total_prompt_tokens - cached

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt_tokens": total_prompt_tokens,
        "cached_tokens": cached,
        "fresh_tokens": fresh,
        "script_id": script_id,
        "shot_id": shot_id,
        "call_type": call_type or "gemini_call",
        "note": note,
    }

    # Update in-memory running totals
    _totals["total_requested"] += total_prompt_tokens
    _totals["total_cached"] += cached
    _totals["call_count"] += 1

    # Write to /storage/logs/token-usage.log
    try:
        log_file = _get_log_file_path()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        logger.warning("Failed to write to token-usage log file", exc_info=True)

    logger.info(
        "Gemini [%s] model=%s: prompt_tokens=%d, cached_tokens=%d, fresh_tokens=%d (script_id=%s)",
        call_type or "call",
        model,
        total_prompt_tokens,
        cached,
        fresh,
        script_id,
    )

    return entry


def get_summary(script_id: Optional[str] = None) -> dict:
    """Calculate total tokens vs cached tokens for a specific script or overall.

    Reads from /storage/logs/token-usage.log and calculates total prompt tokens,
    cached tokens, fresh tokens, savings percentage, and call count.

    Args:
        script_id: Optional script ID string to filter logs by.

    Returns:
        Dict with summary statistics:
        {
            "script_id": str | None,
            "total_prompt_tokens": int,
            "cached_tokens": int,
            "fresh_tokens": int,
            "savings_percentage": float,
            "call_count": int,
        }
    """
    log_file = _get_log_file_path()
    if not log_file.exists():
        return {
            "script_id": script_id,
            "total_prompt_tokens": 0,
            "cached_tokens": 0,
            "fresh_tokens": 0,
            "savings_percentage": 0.0,
            "call_count": 0,
        }

    total_prompt = 0
    total_cached = 0
    count = 0

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if script_id is not None and entry.get("script_id") != str(script_id):
                        continue
                    total_prompt += entry.get("prompt_tokens", 0)
                    total_cached += entry.get("cached_tokens", 0)
                    count += 1
                except json.JSONDecodeError:
                    continue
    except OSError:
        logger.warning("Failed to read token log file", exc_info=True)

    fresh = total_prompt - total_cached
    savings_pct = round((total_cached / total_prompt * 100.0), 2) if total_prompt > 0 else 0.0

    return {
        "script_id": script_id,
        "total_prompt_tokens": total_prompt,
        "cached_tokens": total_cached,
        "fresh_tokens": fresh,
        "savings_percentage": savings_pct,
        "call_count": count,
    }


def get_totals() -> dict:
    """Return running in-memory token totals (backwards compatibility)."""
    total_req = _totals["total_requested"]
    total_cached = _totals["total_cached"]
    return {
        "total_requested": total_req,
        "total_cached": total_cached,
        "total_fresh": total_req - total_cached,
        "call_count": _totals["call_count"],
        "cache_hit_rate": round(total_cached / total_req, 4) if total_req > 0 else 0.0,
    }


def reset_totals() -> None:
    """Reset the running totals (e.g. between test runs)."""
    _totals["total_requested"] = 0
    _totals["total_cached"] = 0
    _totals["call_count"] = 0
