"""
Market Divergence Scoring — World Cup 2026 Win Probability Engine

Computes signed divergence between model win probability and
Polymarket implied probability, normalized by simulation std error.
"""
from __future__ import annotations
import math
from typing import Optional


DIVERGENCE_HIGH_THRESHOLD = 1.5
DIVERGENCE_LOW_THRESHOLD = -1.5
CI_CONFIDENCE = 0.90


def compute_divergence(
    p_model: float,
    p_market: float,
    sigma_model: float,
) -> float:
    """
    divergence = (P_model - P_market) / sigma_model
    +1.5 → model says underpriced (buy signal)
    -1.5 → model says overpriced
    """
    if sigma_model == 0:
        return 0.0
    return round((p_model - p_market) / sigma_model, 3)


def market_outside_ci(p_market: float, ci_lower: float, ci_upper: float) -> bool:
    """True if market odds fall outside the model 90% CI — high conviction divergence."""
    return p_market < ci_lower or p_market > ci_upper


def label_divergence(score: float) -> str:
    if score >= DIVERGENCE_HIGH_THRESHOLD:
        return "underpriced"
    elif score <= DIVERGENCE_LOW_THRESHOLD:
        return "overpriced"
    return "aligned"


def build_divergence_table(
    sim_results: dict[str, dict],
    market_odds: dict[str, float],
) -> list[dict]:
    """
    Merge simulation results with market odds to produce divergence table.

    sim_results: {team: {win_probability, ci_90_lower, ci_90_upper, std, ...}}
    market_odds: {team: implied_probability}

    Returns list of dicts sorted by |divergence_score| descending.
    """
    rows = []
    for team, stats in sim_results.items():
        p_model = stats.get("win_probability", 0.0)
        sigma = stats.get("std", 0.01) or 0.01
        ci_lower = stats.get("ci_90_lower", p_model - 0.02)
        ci_upper = stats.get("ci_90_upper", p_model + 0.02)

        p_market = market_odds.get(team)
        if p_market is None:
            continue

        div_score = compute_divergence(p_model, p_market, sigma)
        rows.append({
            "team": team,
            "model_probability": p_model,
            "market_probability": round(p_market, 4),
            "divergence_score": div_score,
            "divergence_label": label_divergence(div_score),
            "ci_90_lower": ci_lower,
            "ci_90_upper": ci_upper,
            "market_outside_ci": market_outside_ci(p_market, ci_lower, ci_upper),
        })

    return sorted(rows, key=lambda r: abs(r["divergence_score"]), reverse=True)
