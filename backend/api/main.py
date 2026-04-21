"""
FastAPI REST Layer — World Cup 2026 Win Probability Engine
ZerveHack Submission | April 2026

Endpoints:
  GET  /probabilities          — all 48 teams with CI
  GET  /probabilities/{team}   — single team detail
  GET  /divergence             — ranked by market divergence
  POST /bracket/simulate       — single bracket simulation
  GET  /odds/live              — current Polymarket snapshot
"""
from __future__ import annotations
import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Internal imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fbref_loader import load_matches, filter_since
from data.polymarket import get_odds_cached, _get_redis
from engine.elo import ELOEngine
from engine.simulation import SimulationEngine
from engine.divergence import build_divergence_table

logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="World Cup 2026 Win Probability Engine",
    description="ELO ratings + Monte Carlo simulation vs Polymarket odds divergence",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State (computed at startup) ────────────────────────────────────────────────

_state: dict = {
    "sim_results": None,
    "elo_ratings": None,
    "teams": None,
    "groups": None,
    "computed_at": None,
}

_redis = None


@app.on_event("startup")
async def startup():
    global _redis
    _redis = _get_redis()
    await asyncio.get_event_loop().run_in_executor(None, _initialize_engine)


def _initialize_engine():
    """Load data, compute ELO, run simulations. Called once at startup."""
    logger.info("Initializing engine...")
    t0 = time.time()

    df = load_matches()
    df = filter_since(df, 2004)

    elo = ELOEngine()
    elo.process_dataframe(df)

    teams = list(elo.ratings.keys())
    ratings = elo.ratings

    engine = SimulationEngine(teams=teams, ratings=ratings, n_sims=500)
    sim_results = engine.run_with_confidence_intervals()

    _state["sim_results"] = sim_results
    _state["elo_ratings"] = ratings
    _state["teams"] = teams
    _state["groups"] = engine.groups
    _state["computed_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(f"Engine initialized in {time.time() - t0:.1f}s — {len(teams)} teams")


def _require_state():
    if _state["sim_results"] is None:
        raise HTTPException(status_code=503, detail="Engine still initializing. Retry in a moment.")
    return _state


# ── Rate Limiting (simple per-IP) ─────────────────────────────────────────────

_rate_limits: dict[str, list[float]] = {}
RATE_LIMIT = 60  # requests per minute


def _check_rate_limit(ip: str):
    now = time.time()
    window = _rate_limits.get(ip, [])
    window = [t for t in window if now - t < 60]
    if len(window) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 60 req/min.")
    window.append(now)
    _rate_limits[ip] = window


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    try:
        _check_rate_limit(ip)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    return await call_next(request)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "app": "World Cup 2026 Win Probability Engine",
        "hackathon": "ZerveHack 2026",
        "docs": "/docs",
        "endpoints": ["/probabilities", "/divergence", "/bracket/simulate", "/odds/live"],
    }


@app.get("/probabilities")
def get_all_probabilities():
    """Returns win probability for all 48 teams with confidence intervals."""
    state = _require_state()
    odds = get_odds_cached(_redis)
    div_table = {r["team"]: r for r in build_divergence_table(state["sim_results"], odds)}

    teams_out = []
    for team, stats in state["sim_results"].items():
        elo = state["elo_ratings"].get(team, 1500)
        row = {
            "team": team,
            "elo": round(elo, 1),
            **stats,
            "market_probability": odds.get(team),
            "divergence_score": div_table.get(team, {}).get("divergence_score"),
        }
        teams_out.append(row)

    teams_out.sort(key=lambda x: x["win_probability"], reverse=True)

    return {
        "updated_at": state["computed_at"],
        "simulations": 500,
        "teams": teams_out,
    }


@app.get("/probabilities/{team}")
def get_team_probability(team: str):
    """Single team probability with full simulation distribution."""
    state = _require_state()
    # Case-insensitive match
    match = next((t for t in state["sim_results"] if t.lower() == team.lower()), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Team '{team}' not found.")

    odds = get_odds_cached(_redis)
    stats = state["sim_results"][match]
    elo = state["elo_ratings"].get(match, 1500)

    return {
        "team": match,
        "elo": round(elo, 1),
        **stats,
        "market_probability": odds.get(match),
    }


@app.get("/divergence")
def get_divergence(min_score: float = 0.0):
    """Teams ranked by market divergence score (absolute value), descending."""
    state = _require_state()
    odds = get_odds_cached(_redis)
    table = build_divergence_table(state["sim_results"], odds)
    if min_score > 0:
        table = [r for r in table if abs(r["divergence_score"]) >= min_score]
    return {
        "updated_at": state["computed_at"],
        "count": len(table),
        "teams": table,
    }


class BracketSimRequest(BaseModel):
    locked_advances: Optional[dict[str, list[str]]] = None  # stage -> forced teams


@app.post("/bracket/simulate")
def simulate_bracket(req: BracketSimRequest):
    """Run a single bracket simulation and return the full path."""
    state = _require_state()
    engine = SimulationEngine(
        teams=state["teams"],
        ratings=state["elo_ratings"],
        groups=state["groups"],
        n_sims=1,
    )
    result = engine._simulate_once()
    return {
        "quarterfinals": result.get("quarter", []),
        "semifinals": result.get("semi", []),
        "final": result.get("final", []),
        "winner": result.get("winner", []),
    }


@app.get("/odds/live")
def get_live_odds():
    """Current Polymarket odds snapshot (15-min cache)."""
    odds = get_odds_cached(_redis)
    return {
        "source": "Polymarket (gamma-api.polymarket.com)",
        "cache_ttl_seconds": 900,
        "team_count": len(odds),
        "odds": odds,
    }


@app.get("/health")
def health():
    return {"status": "ok", "engine_ready": _state["sim_results"] is not None}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
