"""
World Cup 2026 Win Probability Engine — FastAPI deployment for Zerve
ZerveHack 2026 Submission | Team: Nuriy Gold | hello@nuriy.com
"""
import math, json, datetime, random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI(
    title="WC2026 Win Probability Engine",
    description="ELO + Monte Carlo + Polymarket divergence scoring for World Cup 2026",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "n_sims": 10_000, "k_base": 20, "k_worldcup": 40,
    "home_advantage": 75, "starting_elo": 1500,
    "decay_halflife": 24, "divergence_threshold": 1.5,
}

TEAMS = [
    "Brazil","Argentina","France","England","Spain","Germany","Portugal","Netherlands",
    "Belgium","Italy","Croatia","Denmark","Mexico","USA","Canada","Ecuador","Uruguay",
    "Colombia","Morocco","Senegal","Nigeria","Ghana","Cameroon","Egypt","Japan",
    "South Korea","Australia","Saudi Arabia","Iran","Qatar","Serbia","Switzerland",
    "Poland","Czech Republic","Ukraine","Austria","Costa Rica","Panama","Jamaica",
    "Honduras","Bolivia","Venezuela","Algeria","Tunisia","Mali","DR Congo",
    "South Africa","Ivory Coast",
]

GROUPS = [TEAMS[i*4:(i+1)*4] for i in range(12)]

# ── Synthetic match data + ELO ────────────────────────────────────────────────
def generate_elo_ratings():
    rng = np.random.default_rng(42)
    ratings = {t: 1500.0 for t in TEAMS}
    # Tier adjustments
    tier1 = ["Brazil","Argentina","France","England","Spain","Germany","Portugal","Netherlands"]
    tier2 = ["Belgium","Italy","Croatia","Denmark","Mexico","Morocco","Uruguay","Colombia","Japan","USA"]
    for t in tier1: ratings[t] += rng.uniform(150, 300)
    for t in tier2: ratings[t] += rng.uniform(50, 150)
    return ratings

_elo_cache = None
def get_elo():
    global _elo_cache
    if _elo_cache is None:
        _elo_cache = generate_elo_ratings()
    return _elo_cache

# ── Match simulation ──────────────────────────────────────────────────────────
def elo_win_prob(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def simulate_match(team_a, team_b, elo_ratings, neutral=True):
    ea = elo_ratings.get(team_a, 1500)
    eb = elo_ratings.get(team_b, 1500)
    if not neutral: ea += CONFIG["home_advantage"]
    p_a = elo_win_prob(ea, eb)
    lam_a = max(0.3, 1.5 * p_a / 0.5)
    lam_b = max(0.3, 1.5 * (1 - p_a) / 0.5)
    goals_a = np.random.poisson(lam_a)
    goals_b = np.random.poisson(lam_b)
    if goals_a > goals_b: return team_a
    elif goals_b > goals_a: return team_b
    else: return team_a if np.random.random() < p_a else team_b

# ── Monte Carlo ───────────────────────────────────────────────────────────────
def simulate_tournament(elo_ratings):
    wins = {t: 0 for t in TEAMS}
    # Group stage
    qualified = []
    for group in GROUPS:
        pts = {t: 0 for t in group}
        for i, ta in enumerate(group):
            for tb in group[i+1:]:
                w = simulate_match(ta, tb, elo_ratings)
                pts[w] += 3
        sorted_group = sorted(pts, key=lambda t: pts[t], reverse=True)
        qualified.extend(sorted_group[:2])
    # Fill to 32 with best 3rd place (simplified)
    third = []
    for group in GROUPS:
        pts = {t: 0 for t in group}
        for i, ta in enumerate(group):
            for tb in group[i+1:]:
                w = simulate_match(ta, tb, elo_ratings)
                pts[w] += 3
        sorted_group = sorted(pts, key=lambda t: pts[t], reverse=True)
        third.append((sorted_group[2], pts[sorted_group[2]]))
    third.sort(key=lambda x: x[1], reverse=True)
    qualified.extend([t for t, _ in third[:8]])
    # Knockout
    bracket = qualified[:32]
    np.random.shuffle(bracket)
    while len(bracket) > 1:
        next_round = []
        for i in range(0, len(bracket), 2):
            if i+1 < len(bracket):
                w = simulate_match(bracket[i], bracket[i+1], elo_ratings)
                next_round.append(w)
        bracket = next_round
    if bracket: wins[bracket[0]] += 1
    return wins

_sim_cache = None
_sim_timestamp = None

def get_simulation_results():
    global _sim_cache, _sim_timestamp
    now = datetime.datetime.utcnow()
    if _sim_cache and _sim_timestamp and (now - _sim_timestamp).seconds < 3600:
        return _sim_cache
    elo = get_elo()
    n = CONFIG["n_sims"]
    win_counts = {t: 0 for t in TEAMS}
    for _ in range(n):
        result = simulate_tournament(elo)
        for t, w in result.items():
            win_counts[t] += w
    probs = {t: win_counts[t] / n for t in TEAMS}
    sigma = {t: math.sqrt(probs[t] * (1 - probs[t]) / n) for t in TEAMS}
    _sim_cache = {"probs": probs, "sigma": sigma, "win_counts": win_counts}
    _sim_timestamp = now
    return _sim_cache

def get_market_odds():
    rng = np.random.default_rng(99)
    elo = get_elo()
    raw = {t: max(0.001, elo[t] - 1400 + rng.normal(0, 30)) for t in TEAMS}
    total = sum(raw.values())
    return {t: raw[t] / total for t in TEAMS}

# ── Startup: pre-warm (run sims in background on startup) ────────────────────
@app.on_event("startup")
async def startup():
    import threading
    def warm(): get_simulation_results()
    threading.Thread(target=warm, daemon=True).start()

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "WC2026 Win Probability Engine",
        "team": "Nuriy Gold",
        "endpoints": ["/probabilities", "/divergence", "/bracket/simulate", "/odds/live"],
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()}

@app.get("/probabilities")
def probabilities():
    sim = get_simulation_results()
    elo = get_elo()
    market = get_market_odds()
    z = 1.645  # 90% CI
    results = []
    for team in TEAMS:
        p = sim["probs"][team]
        s = sim["sigma"][team]
        market_p = market.get(team, 0)
        div = (p - market_p) / s if s > 0 else 0
        results.append({
            "team": team,
            "elo": round(elo.get(team, 1500), 1),
            "win_probability": round(p, 4),
            "ci_low": round(max(0, p - z * s), 4),
            "ci_high": round(min(1, p + z * s), 4),
            "market_probability": round(market_p, 4),
            "divergence_score": round(div, 3),
            "signal": "underpriced" if div > CONFIG["divergence_threshold"] else "overpriced" if div < -CONFIG["divergence_threshold"] else "fair",
        })
    return {"teams": sorted(results, key=lambda x: x["win_probability"], reverse=True),
            "updated_at": datetime.datetime.utcnow().isoformat()}

@app.get("/probabilities/{team}")
def team_probability(team: str):
    sim = get_simulation_results()
    elo = get_elo()
    market = get_market_odds()
    team_key = next((t for t in TEAMS if t.lower() == team.lower()), None)
    if not team_key:
        return {"error": f"Team '{team}' not found"}
    p = sim["probs"][team_key]
    s = sim["sigma"][team_key]
    market_p = market.get(team_key, 0)
    div = (p - market_p) / s if s > 0 else 0
    return {
        "team": team_key,
        "elo": round(elo.get(team_key, 1500), 1),
        "win_probability": round(p, 4),
        "market_probability": round(market_p, 4),
        "divergence_score": round(div, 3),
        "signal": "underpriced" if div > CONFIG["divergence_threshold"] else "overpriced" if div < -CONFIG["divergence_threshold"] else "fair",
    }

@app.get("/divergence")
def divergence():
    sim = get_simulation_results()
    market = get_market_odds()
    rows = []
    for team in TEAMS:
        p = sim["probs"][team]
        s = sim["sigma"][team]
        market_p = market.get(team, 0)
        div = (p - market_p) / s if s > 0 else 0
        rows.append({
            "team": team,
            "model_prob": round(p, 4),
            "market_prob": round(market_p, 4),
            "divergence_score": round(div, 3),
            "signal": "underpriced" if div > CONFIG["divergence_threshold"] else "overpriced" if div < -CONFIG["divergence_threshold"] else "fair",
            "outside_ci": abs(div) > CONFIG["divergence_threshold"],
        })
    return {"divergence": sorted(rows, key=lambda x: abs(x["divergence_score"]), reverse=True),
            "threshold": CONFIG["divergence_threshold"]}

@app.post("/bracket/simulate")
def bracket_simulate():
    elo = get_elo()
    result = simulate_tournament(elo)
    winner = max(result, key=lambda t: result[t])
    return {"winner": winner, "result": result,
            "timestamp": datetime.datetime.utcnow().isoformat()}

@app.get("/odds/live")
def odds_live():
    market = get_market_odds()
    return {"odds": {t: round(v, 4) for t, v in sorted(market.items(), key=lambda x: x[1], reverse=True)},
            "source": "Polymarket (synthetic — real data requires POLYMARKET_API_KEY)",
            "cached_at": datetime.datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
