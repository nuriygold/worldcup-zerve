"""
Polymarket Odds Loader — World Cup 2026 Win Probability Engine

Fetches live implied probabilities from Polymarket REST API.
All calls are proxied through backend (key never exposed to client).
Redis-cached with 15-minute TTL.
Falls back to synthetic odds in hackathon/demo mode.
"""
from __future__ import annotations
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

POLYMARKET_API_BASE = "https://gamma-api.polymarket.com"
CACHE_TTL = 900  # 15 minutes in seconds

# In-memory cache fallback when Redis not available
_memory_cache: dict = {"data": None, "expires_at": 0.0}


def _get_redis():
    """Try to get Redis client; return None if unavailable."""
    try:
        import redis
        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            socket_connect_timeout=1,
        )
        r.ping()
        return r
    except Exception:
        return None


def fetch_live_odds(market_slug: str = "fifa-world-cup-2026-winner") -> Optional[dict[str, float]]:
    """
    Fetch World Cup winner market from Polymarket.
    Returns {team_name: implied_probability} or None on failure.
    """
    import httpx
    try:
        url = f"{POLYMARKET_API_BASE}/markets?slug={market_slug}"
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        odds = {}
        for market in data if isinstance(data, list) else [data]:
            for outcome in market.get("outcomes", []):
                label = outcome.get("label") or outcome.get("name", "")
                prob = float(outcome.get("probability", outcome.get("price", 0)))
                if label:
                    odds[label] = round(prob, 4)
        return odds if odds else None
    except Exception as e:
        logger.warning(f"Polymarket API unavailable: {e}")
        return None


def get_odds_cached(redis_client=None) -> dict[str, float]:
    """
    Return cached odds. Checks Redis first, then memory cache, then fetches live.
    Falls back to synthetic odds.
    """
    cache_key = "worldcup:polymarket_odds"

    # Try Redis
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

    # Try memory cache
    now = time.time()
    if _memory_cache["data"] and now < _memory_cache["expires_at"]:
        return _memory_cache["data"]

    # Fetch live
    odds = fetch_live_odds()
    if odds:
        if redis_client:
            try:
                redis_client.setex(cache_key, CACHE_TTL, json.dumps(odds))
            except Exception:
                pass
        _memory_cache["data"] = odds
        _memory_cache["expires_at"] = now + CACHE_TTL
        return odds

    # Synthetic fallback for demo
    return _synthetic_odds()


def _synthetic_odds() -> dict[str, float]:
    """
    Realistic synthetic Polymarket-style odds for all 48 WC teams.
    Based on approximate real market expectations as of April 2026.
    """
    raw = {
        "Brazil": 0.155, "Argentina": 0.165, "France": 0.145, "England": 0.115,
        "Spain": 0.090, "Germany": 0.075, "Portugal": 0.060, "Netherlands": 0.040,
        "Belgium": 0.018, "Italy": 0.020, "Croatia": 0.012, "Denmark": 0.010,
        "Mexico": 0.014, "USA": 0.016, "Canada": 0.008, "Ecuador": 0.005,
        "Uruguay": 0.012, "Colombia": 0.009, "Morocco": 0.011, "Senegal": 0.006,
        "Nigeria": 0.004, "Ghana": 0.003, "Cameroon": 0.003, "Egypt": 0.003,
        "Japan": 0.008, "South Korea": 0.005, "Australia": 0.003, "Saudi Arabia": 0.002,
        "Iran": 0.002, "Qatar": 0.001, "Serbia": 0.006, "Switzerland": 0.007,
        "Poland": 0.004, "Czech Republic": 0.003, "Ukraine": 0.004, "Austria": 0.003,
        "Costa Rica": 0.002, "Panama": 0.001, "Jamaica": 0.001, "Honduras": 0.001,
        "Bolivia": 0.001, "Venezuela": 0.001, "Algeria": 0.002, "Tunisia": 0.002,
        "Mali": 0.001, "DR Congo": 0.001, "South Africa": 0.002, "Ivory Coast": 0.003,
    }
    # Normalize to sum to 1
    total = sum(raw.values())
    return {k: round(v / total, 4) for k, v in raw.items()}
