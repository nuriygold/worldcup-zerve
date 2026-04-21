"""
ELO Rating Engine — World Cup 2026 Win Probability Engine
Follows World Football ELO standard with goal margin scaling,
home advantage, and exponential time decay.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────
K_BASE = 20
K_WORLDCUP = 40
HOME_ADVANTAGE = 75      # ELO points added for home team
STARTING_RATING = 1500
DECAY_HALFLIFE_MONTHS = 24  # months for exponential decay


def k_factor(competition: str) -> float:
    """Return K factor based on competition type."""
    comp = competition.lower()
    if "world cup" in comp and "qualifier" not in comp:
        return K_WORLDCUP
    if "confederations" in comp or "continental" in comp or "nations" in comp:
        return 30
    return K_BASE


def goal_margin_multiplier(home_goals: int, away_goals: int) -> float:
    """Diminishing-returns scaling on goal margin."""
    diff = abs(home_goals - away_goals)
    return math.log(diff + 1) + 1.0


def expected_score(rating_own: float, rating_opp: float) -> float:
    """Standard ELO expected score."""
    return 1.0 / (1.0 + 10 ** ((rating_opp - rating_own) / 400.0))


def actual_score(home_goals: int, away_goals: int, perspective: str = "home") -> float:
    """Return 1.0/0.5/0.0 for win/draw/loss from the given perspective."""
    if home_goals > away_goals:
        return 1.0 if perspective == "home" else 0.0
    elif home_goals == away_goals:
        return 0.5
    else:
        return 0.0 if perspective == "home" else 1.0


def decay_weight(match_date: date, reference_date: Optional[date] = None) -> float:
    """Exponential decay weight based on months since match."""
    if reference_date is None:
        reference_date = date.today()
    delta_months = (
        (reference_date.year - match_date.year) * 12
        + (reference_date.month - match_date.month)
    )
    lam = math.log(2) / DECAY_HALFLIFE_MONTHS
    return math.exp(-lam * max(delta_months, 0))


@dataclass
class ELOEngine:
    """
    Stateful ELO engine. Feed matches in chronological order.
    Ratings are stored per team ISO code.
    """
    ratings: dict[str, float] = field(default_factory=dict)
    match_count: dict[str, int] = field(default_factory=dict)

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, STARTING_RATING)

    def process_match(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        competition: str,
        neutral: bool = False,
        match_date: Optional[date] = None,
        reference_date: Optional[date] = None,
    ) -> tuple[float, float]:
        """
        Update ratings for one match. Returns (new_home_rating, new_away_rating).
        Applies time-decay weighting to K factor if match_date provided.
        """
        r_home = self.get_rating(home)
        r_away = self.get_rating(away)

        # Home advantage only applies on non-neutral ground
        effective_home = r_home + (0 if neutral else HOME_ADVANTAGE)

        exp_home = expected_score(effective_home, r_away)
        exp_away = 1.0 - exp_home

        act_home = actual_score(home_goals, away_goals, "home")
        act_away = actual_score(home_goals, away_goals, "away")

        k = k_factor(competition)
        gm = goal_margin_multiplier(home_goals, away_goals)

        # Apply time decay to K if date info available
        w = 1.0
        if match_date:
            w = decay_weight(match_date, reference_date)

        k_eff = k * gm * w

        new_home = r_home + k_eff * (act_home - exp_home)
        new_away = r_away + k_eff * (act_away - exp_away)

        self.ratings[home] = new_home
        self.ratings[away] = new_away
        self.match_count[home] = self.match_count.get(home, 0) + 1
        self.match_count[away] = self.match_count.get(away, 0) + 1

        return new_home, new_away

    def process_dataframe(self, df) -> None:
        """
        Process a pandas DataFrame with columns:
        date, home_team, away_team, home_goals, away_goals, competition, neutral
        Assumes df is sorted chronologically.
        """
        import pandas as pd
        reference = date.today()
        for _, row in df.iterrows():
            match_date = row["date"]
            if isinstance(match_date, str):
                match_date = date.fromisoformat(match_date)
            elif hasattr(match_date, "date"):
                match_date = match_date.date()
            self.process_match(
                home=row["home_team"],
                away=row["away_team"],
                home_goals=int(row["home_goals"]),
                away_goals=int(row["away_goals"]),
                competition=str(row.get("competition", "Friendly")),
                neutral=bool(row.get("neutral", False)),
                match_date=match_date,
                reference_date=reference,
            )

    def snapshot(self) -> list[dict]:
        """Return current ratings sorted descending."""
        return sorted(
            [{"team": t, "elo": round(r, 1), "matches": self.match_count.get(t, 0)}
             for t, r in self.ratings.items()],
            key=lambda x: x["elo"],
            reverse=True,
        )
