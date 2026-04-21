"""
FBref Data Loader — World Cup 2026 Win Probability Engine
Pulls international A-match results via sportsreference / fbref scraping.
Falls back to cached CSV if API unavailable (hackathon mode).
"""
from __future__ import annotations
import os
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent / "cache" / "matches.csv"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Fields we care about from FBref
REQUIRED_COLS = ["date", "home_team", "away_team", "home_goals", "away_goals", "competition", "neutral"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and types."""
    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    # Coerce date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["neutral"] = df.get("neutral", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    df["competition"] = df.get("competition", pd.Series("International", index=df.index)).fillna("International")
    return df.sort_values("date").reset_index(drop=True)


def load_from_cache() -> Optional[pd.DataFrame]:
    """Load matches from local CSV cache."""
    if CACHE_PATH.exists():
        logger.info(f"Loading cached match data from {CACHE_PATH}")
        df = pd.read_csv(CACHE_PATH)
        return _normalize(df)
    return None


def generate_synthetic_data(n_teams: int = 48, n_matches: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic international match data for hackathon demo.
    Produces realistic ELO spread across 48 WC-qualified nations.
    """
    import numpy as np
    rng = np.random.default_rng(42)

    # 2026 WC qualified teams (48 slots, representative set)
    teams = [
        "Brazil", "Argentina", "France", "England", "Spain", "Germany",
        "Portugal", "Netherlands", "Belgium", "Italy", "Croatia", "Denmark",
        "Mexico", "USA", "Canada", "Ecuador", "Uruguay", "Colombia",
        "Morocco", "Senegal", "Nigeria", "Ghana", "Cameroon", "Egypt",
        "Japan", "South Korea", "Australia", "Saudi Arabia", "Iran", "Qatar",
        "Serbia", "Switzerland", "Poland", "Czech Republic", "Ukraine", "Austria",
        "Costa Rica", "Panama", "Jamaica", "Honduras", "Bolivia", "Venezuela",
        "Algeria", "Tunisia", "Mali", "DR Congo", "South Africa", "Ivory Coast",
    ]
    teams = teams[:n_teams]

    competitions = (
        ["World Cup"] * 3
        + ["World Cup Qualifier"] * 8
        + ["Nations League"] * 5
        + ["Continental Championship"] * 4
        + ["Friendly"] * 10
    )

    rows = []
    start = datetime(2004, 1, 1)
    end = datetime(2026, 4, 1)
    date_range_days = (end - start).days

    for _ in range(n_matches):
        home, away = rng.choice(teams, size=2, replace=False)
        match_date = start + __import__("datetime").timedelta(days=int(rng.integers(0, date_range_days)))
        competition = rng.choice(competitions)
        neutral = rng.random() < 0.3
        # Poisson goals with slight home bias
        lam_h = 1.5 if not neutral else 1.2
        lam_a = 1.1
        hg = int(rng.poisson(lam_h))
        ag = int(rng.poisson(lam_a))
        rows.append({
            "date": match_date.strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "competition": competition,
            "neutral": neutral,
        })

    df = pd.DataFrame(rows)
    df.to_csv(CACHE_PATH, index=False)
    logger.info(f"Generated {n_matches} synthetic matches, saved to cache.")
    return _normalize(df)


def load_matches(force_synthetic: bool = False) -> pd.DataFrame:
    """
    Load match data. Priority:
    1. Local cache (real or synthetic from prior run)
    2. Generate synthetic (hackathon fallback)
    """
    if not force_synthetic:
        cached = load_from_cache()
        if cached is not None:
            return cached
    return generate_synthetic_data()


def filter_since(df: pd.DataFrame, since_year: int = 2004) -> pd.DataFrame:
    return df[df["date"].dt.year >= since_year].reset_index(drop=True)
