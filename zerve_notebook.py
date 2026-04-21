"""
# ⚽ World Cup 2026 Win Probability Engine
## ZerveHack 2026 Submission

**Stack:** Python · NumPy · pandas · Plotly · Zerve Deploy

This notebook:
1. Generates 20yr international match history (FBref-structured)
2. Computes ELO ratings with time decay + goal margin scaling
3. Runs 10,000 Monte Carlo tournament simulations (Dixon-Coles Poisson)
4. Cross-validates against Polymarket implied odds
5. Scores market divergence per team: (P_model - P_market) / σ
6. Renders interactive Plotly charts

Deploy this notebook via Zerve → your endpoints go live instantly.
"""

# ── Block 1: Imports ──────────────────────────────────────────────────────────
import math
import json
import datetime
import numpy as np
import pandas as pd

print("✅ Imports loaded")

# ── Block 2: Configuration ────────────────────────────────────────────────────
CONFIG = {
    "n_sims": 10_000,          # Monte Carlo simulations
    "k_base": 20,              # ELO K factor — regular internationals
    "k_worldcup": 40,          # ELO K factor — World Cup matches
    "home_advantage": 75,      # ELO points for home team
    "starting_elo": 1500,      # Default rating for all teams
    "decay_halflife": 24,      # Months for exponential time decay
    "divergence_threshold": 1.5,  # σ cutoff for underpriced/overpriced signal
}

# 2026 World Cup — 48 qualified teams
TEAMS = [
    "Brazil", "Argentina", "France", "England", "Spain", "Germany",
    "Portugal", "Netherlands", "Belgium", "Italy", "Croatia", "Denmark",
    "Mexico", "USA", "Canada", "Ecuador", "Uruguay", "Colombia",
    "Morocco", "Senegal", "Nigeria", "Ghana", "Cameroon", "Egypt",
    "Japan", "South Korea", "Australia", "Saudi Arabia", "Iran", "Qatar",
    "Serbia", "Switzerland", "Poland", "Czech Republic", "Ukraine", "Austria",
    "Costa Rica", "Panama", "Jamaica", "Honduras", "Bolivia", "Venezuela",
    "Algeria", "Tunisia", "Mali", "DR Congo", "South Africa", "Ivory Coast",
]

print(f"✅ Config loaded — {len(TEAMS)} teams, {CONFIG['n_sims']:,} simulations")

# ── Block 3: Synthetic Match Data (FBref-structured) ─────────────────────────
def generate_match_data(teams, n_matches=6000, seed=42):
    """
    Generate 20 years of international A-match results.
    Structure mirrors FBref: date, home_team, away_team, goals, competition, neutral.
    Replace with real FBref pull via sportsreference in production.
    """
    rng = np.random.default_rng(seed)
    competitions = (
        ["World Cup"] * 3 + ["World Cup Qualifier"] * 8 +
        ["Nations League"] * 5 + ["Continental Championship"] * 4 +
        ["Friendly"] * 10
    )
    start = datetime.datetime(2004, 1, 1)
    rows = []
    for _ in range(n_matches):
        home, away = rng.choice(teams, size=2, replace=False)
        days = int(rng.integers(0, (datetime.datetime(2026, 4, 1) - start).days))
        match_date = start + datetime.timedelta(days=days)
        comp = rng.choice(competitions)
        neutral = rng.random() < 0.3
        hg = int(rng.poisson(1.5 if not neutral else 1.2))
        ag = int(rng.poisson(1.1))
        rows.append({
            "date": match_date,
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "competition": comp,
            "neutral": neutral,
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df

matches = generate_match_data(TEAMS)
print(f"✅ Match data: {len(matches):,} matches across {matches['home_team'].nunique()} teams")
print(f"   Date range: {matches['date'].min().date()} → {matches['date'].max().date()}")

# ── Block 4: ELO Engine ───────────────────────────────────────────────────────
def k_factor(competition):
    comp = competition.lower()
    if "world cup" in comp and "qualifier" not in comp:
        return CONFIG["k_worldcup"]
    if any(x in comp for x in ["nations", "continental", "confederation"]):
        return 30
    return CONFIG["k_base"]

def decay_weight(match_date, ref=None):
    if ref is None:
        ref = datetime.date.today()
    if hasattr(match_date, "date"):
        match_date = match_date.date()
    months_ago = (ref.year - match_date.year) * 12 + (ref.month - match_date.month)
    lam = math.log(2) / CONFIG["decay_halflife"]
    return math.exp(-lam * max(months_ago, 0))

def compute_elo_ratings(df):
    ratings = {t: CONFIG["starting_elo"] for t in TEAMS}
    ref = datetime.date.today()

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        comp = str(row["competition"])
        neutral = bool(row["neutral"])

        r_home = ratings.get(home, CONFIG["starting_elo"])
        r_away = ratings.get(away, CONFIG["starting_elo"])

        # Home advantage (none on neutral ground)
        eff_home = r_home + (0 if neutral else CONFIG["home_advantage"])

        # Expected scores
        exp_home = 1.0 / (1.0 + 10 ** ((r_away - eff_home) / 400.0))
        exp_away = 1.0 - exp_home

        # Actual scores
        if hg > ag:
            act_home, act_away = 1.0, 0.0
        elif hg == ag:
            act_home, act_away = 0.5, 0.5
        else:
            act_home, act_away = 0.0, 1.0

        # Goal margin scaling + time decay
        gm = math.log(abs(hg - ag) + 1) + 1.0
        k = k_factor(comp)
        w = decay_weight(row["date"], ref)
        k_eff = k * gm * w

        ratings[home] = r_home + k_eff * (act_home - exp_home)
        ratings[away] = r_away + k_eff * (act_away - exp_away)

    return ratings

elo_ratings = compute_elo_ratings(matches)
top5 = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"\n✅ ELO computed for {len(elo_ratings)} teams")
print("   Top 5:")
for team, elo in top5:
    print(f"   {team}: {elo:.1f}")

# ── Block 5: Monte Carlo Simulation ──────────────────────────────────────────
def match_outcome_probs(elo_home, elo_away, neutral=True):
    """Dixon-Coles adjusted Poisson — returns (p_home_win, p_draw, p_away_win)."""
    diff = elo_home - elo_away
    if not neutral:
        diff += CONFIG["home_advantage"]
    lam_h = 1.35 * math.exp(diff / 800.0)
    lam_a = 1.10 * math.exp(-diff / 800.0)

    MAX_G = 8
    rho = -0.13  # Dixon-Coles correlation

    def dc_tau(h, a):
        if h == 0 and a == 0: return 1 - lam_h * lam_a * rho
        if h == 0 and a == 1: return 1 + lam_h * rho
        if h == 1 and a == 0: return 1 + lam_a * rho
        if h == 1 and a == 1: return 1 - rho
        return 1.0

    def poisson_pmf(k, lam):
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    ph = pd_ = pa = 0.0
    for h in range(MAX_G + 1):
        for a in range(MAX_G + 1):
            p = poisson_pmf(h, lam_h) * poisson_pmf(a, lam_a) * dc_tau(h, a)
            if h > a: ph += p
            elif h == a: pd_ += p
            else: pa += p

    total = ph + pd_ + pa
    return ph / total, pd_ / total, pa / total

def simulate_group(group, ratings):
    points = {t: 0 for t in group}
    gd = {t: 0 for t in group}
    for i, home in enumerate(group):
        for away in group[i+1:]:
            ph, pd_, pa = match_outcome_probs(
                ratings.get(home, 1500), ratings.get(away, 1500)
            )
            r = np.random.random()
            if r < ph:
                points[home] += 3; gd[home] += 1; gd[away] -= 1
            elif r < ph + pd_:
                points[home] += 1; points[away] += 1
            else:
                points[away] += 3; gd[away] += 1; gd[home] -= 1
    return sorted(group, key=lambda t: (points[t], gd[t]), reverse=True)

def simulate_knockout(teams, ratings):
    winners = []
    for i in range(0, len(teams), 2):
        if i + 1 >= len(teams):
            winners.append(teams[i]); continue
        h, a = teams[i], teams[i+1]
        ph, _, pa = match_outcome_probs(ratings.get(h, 1500), ratings.get(a, 1500))
        r = np.random.random()
        winners.append(h if r < ph else (a if r < ph + pa else
                       (h if np.random.random() < 0.5 else a)))
    return winners

def run_simulations(teams, ratings, n_sims, seed=42):
    np.random.seed(seed)

    # 12 groups of 4 — seeded by ELO
    sorted_teams = sorted(teams, key=lambda t: ratings.get(t, 1500), reverse=True)
    groups = [[] for _ in range(12)]
    for i, t in enumerate(sorted_teams):
        groups[i % 12].append(t)

    wins = {t: 0 for t in teams}
    finals = {t: 0 for t in teams}
    semis = {t: 0 for t in teams}
    quarters = {t: 0 for t in teams}
    win_log = {t: [] for t in teams}

    for sim in range(n_sims):
        r32 = []
        all_thirds = []
        for group in groups:
            standing = simulate_group(group, ratings)
            r32.extend(standing[:2])
            all_thirds.append(standing[2])

        # Best 8 third-place finishers (simplified: first 8)
        r32.extend(all_thirds[:8])

        q = simulate_knockout(r32[:32], ratings)
        s = simulate_knockout(q, ratings)
        f = simulate_knockout(s, ratings)
        w = simulate_knockout(f, ratings)

        for t in q: quarters[t] += 1
        for t in s: semis[t] += 1
        for t in f: finals[t] += 1
        for t in w:
            wins[t] += 1
            win_log[t].append(1)
        for t in teams:
            if t not in w:
                win_log[t].append(0)

        if (sim + 1) % 1000 == 0:
            print(f"   ... {sim+1:,}/{n_sims:,} simulations complete")

    results = {}
    for t in teams:
        arr = np.array(win_log[t], dtype=float)
        mean = arr.mean()
        std = arr.std()
        ci_lo = max(0, mean - 1.645 * std / math.sqrt(n_sims))
        ci_hi = min(1, mean + 1.645 * std / math.sqrt(n_sims))
        results[t] = {
            "win_probability": round(mean, 4),
            "ci_90_lower": round(ci_lo, 4),
            "ci_90_upper": round(ci_hi, 4),
            "std": round(std, 4),
            "finalist_probability": round(finals[t] / n_sims, 4),
            "semifinal_probability": round(semis[t] / n_sims, 4),
            "quarterfinal_probability": round(quarters[t] / n_sims, 4),
        }
    return results

print(f"\n⏳ Running {CONFIG['n_sims']:,} Monte Carlo simulations...")
sim_results = run_simulations(TEAMS, elo_ratings, CONFIG["n_sims"])

top10 = sorted(sim_results.items(), key=lambda x: x[1]["win_probability"], reverse=True)[:10]
print("\n✅ Simulation complete. Top 10 win probabilities:")
for team, stats in top10:
    elo = elo_ratings.get(team, 1500)
    print(f"   {team:20s} {stats['win_probability']*100:5.1f}%  "
          f"(90% CI: {stats['ci_90_lower']*100:.1f}–{stats['ci_90_upper']*100:.1f}%)  "
          f"ELO: {elo:.0f}")

# ── Block 6: Polymarket Odds (live fetch → synthetic fallback) ────────────────
def get_market_odds():
    """
    Fetch live Polymarket World Cup winner odds.
    Falls back to calibrated synthetic odds if API unavailable.
    In Zerve production: connect to Polymarket API with 15-min scheduled refresh.
    """
    try:
        import urllib.request
        url = "https://gamma-api.polymarket.com/markets?slug=fifa-world-cup-2026-winner"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            odds = {}
            for market in (data if isinstance(data, list) else [data]):
                for outcome in market.get("outcomes", []):
                    label = outcome.get("label") or outcome.get("name", "")
                    prob = float(outcome.get("probability", outcome.get("price", 0)))
                    if label:
                        odds[label] = round(prob, 4)
            if odds:
                print("✅ Live Polymarket odds fetched")
                return odds
    except Exception as e:
        print(f"   Polymarket API unavailable ({e}) — using synthetic odds")

    # Calibrated synthetic odds (normalized to sum=1)
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
    total = sum(raw.values())
    return {k: round(v / total, 4) for k, v in raw.items()}

market_odds = get_market_odds()

# ── Block 7: Divergence Scoring ───────────────────────────────────────────────
def compute_divergence_table(sim_results, market_odds):
    rows = []
    for team, stats in sim_results.items():
        p_model = stats["win_probability"]
        sigma = stats["std"] or 0.001
        p_market = market_odds.get(team)
        if p_market is None:
            continue
        div = round((p_model - p_market) / sigma, 3)
        ci_lo = stats["ci_90_lower"]
        ci_hi = stats["ci_90_upper"]
        label = ("underpriced" if div >= CONFIG["divergence_threshold"]
                 else "overpriced" if div <= -CONFIG["divergence_threshold"]
                 else "aligned")
        rows.append({
            "team": team,
            "elo": round(elo_ratings.get(team, 1500), 1),
            "model_prob": p_model,
            "market_prob": p_market,
            "divergence_score": div,
            "signal": label,
            "ci_90_lower": ci_lo,
            "ci_90_upper": ci_hi,
            "market_outside_ci": p_market < ci_lo or p_market > ci_hi,
        })
    return sorted(rows, key=lambda r: abs(r["divergence_score"]), reverse=True)

div_table = compute_divergence_table(sim_results, market_odds)

underpriced = [r for r in div_table if r["signal"] == "underpriced"]
overpriced  = [r for r in div_table if r["signal"] == "overpriced"]

print(f"\n✅ Divergence scoring complete")
print(f"   Underpriced (model >> market): {len(underpriced)} teams")
print(f"   Overpriced  (market >> model): {len(overpriced)} teams")
print(f"\n   Top 5 underpriced:")
for r in underpriced[:5]:
    print(f"   {r['team']:20s}  model={r['model_prob']*100:.1f}%  market={r['market_prob']*100:.1f}%  score=+{r['divergence_score']:.2f}σ")
print(f"\n   Top 5 overpriced:")
for r in overpriced[:5]:
    print(f"   {r['team']:20s}  model={r['model_prob']*100:.1f}%  market={r['market_prob']*100:.1f}%  score={r['divergence_score']:.2f}σ")

# ── Block 8: Visualizations ───────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    top20 = sorted(sim_results.items(), key=lambda x: x[1]["win_probability"], reverse=True)[:20]
    teams_chart = [t for t, _ in top20]
    probs_chart = [s["win_probability"] * 100 for _, s in top20]
    ci_lo_chart = [(s["win_probability"] - s["ci_90_lower"]) * 100 for _, s in top20]
    ci_hi_chart = [(s["ci_90_upper"] - s["win_probability"]) * 100 for _, s in top20]
    market_chart = [market_odds.get(t, 0) * 100 for t in teams_chart]

    # Chart 1: Win probability with CI
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=probs_chart, y=teams_chart, orientation="h",
        name="Model Probability",
        marker_color="#3b82f6",
        error_x=dict(type="data", symmetric=False,
                     array=ci_hi_chart, arrayminus=ci_lo_chart,
                     color="#60a5fa", thickness=2, width=6),
    ))
    fig1.add_trace(go.Bar(
        x=market_chart, y=teams_chart, orientation="h",
        name="Polymarket Odds",
        marker_color="#f59e0b", opacity=0.6,
    ))
    fig1.update_layout(
        title="⚽ WC2026 Win Probabilities: Model vs Market (Top 20)",
        xaxis_title="Win Probability (%)",
        barmode="overlay",
        height=600,
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(autorange="reversed"),
        legend=dict(bgcolor="#1e293b"),
    )
    fig1.show()

    # Chart 2: Divergence scatter
    div_teams = [r["team"] for r in div_table if r["market_prob"] is not None][:30]
    model_vals = [sim_results[t]["win_probability"] * 100 for t in div_teams]
    market_vals = [market_odds.get(t, 0) * 100 for t in div_teams]
    colors = [
        "#22c55e" if sim_results[t]["win_probability"] > market_odds.get(t, 0)
        else "#ef4444"
        for t in div_teams
    ]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=market_vals, y=model_vals, mode="markers+text",
        text=div_teams, textposition="top center",
        textfont=dict(size=9, color="#94a3b8"),
        marker=dict(color=colors, size=10, opacity=0.85),
        hovertemplate="<b>%{text}</b><br>Market: %{x:.1f}%<br>Model: %{y:.1f}%<extra></extra>",
    ))
    # Diagonal y=x
    max_val = max(max(model_vals), max(market_vals)) * 1.1
    fig2.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode="lines",
        line=dict(color="#475569", dash="dot", width=1),
        showlegend=False, hoverinfo="none",
    ))
    fig2.update_layout(
        title="📊 Model vs Market — Divergence Map<br><sup>🟢 = Model sees upside | 🔴 = Market overpricing</sup>",
        xaxis_title="Polymarket Implied Probability (%)",
        yaxis_title="Model Win Probability (%)",
        height=550,
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
    )
    fig2.show()

    print("\n✅ Charts rendered")

except ImportError:
    print("\nPlotly not available — install with: pip install plotly")

# ── Block 9: JSON Output (Zerve API Endpoint) ─────────────────────────────────
"""
When deployed via Zerve, the final cell's return value becomes the API response.
Zerve serves this as a live JSON endpoint at your deployment URL.
Hit /probabilities, /divergence etc. from the React frontend.
"""

api_response = {
    "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
    "simulations": CONFIG["n_sims"],
    "model": {
        "elo_decay_halflife_months": CONFIG["decay_halflife"],
        "k_worldcup": CONFIG["k_worldcup"],
        "k_base": CONFIG["k_base"],
        "home_advantage_elo": CONFIG["home_advantage"],
    },
    "probabilities": sorted([
        {
            "team": team,
            "elo": round(elo_ratings.get(team, 1500), 1),
            **stats,
            "market_probability": market_odds.get(team),
        }
        for team, stats in sim_results.items()
    ], key=lambda x: x["win_probability"], reverse=True),
    "divergence": div_table[:20],
}

# Print summary
top3 = api_response["probabilities"][:3]
print(f"\n🏆 Final output ready for Zerve deployment")
print(f"   Top 3: " + " | ".join(f"{t['team']} {t['win_probability']*100:.1f}%" for t in top3))
print(f"   Strongest signal: {div_table[0]['team']} ({div_table[0]['signal']}, {div_table[0]['divergence_score']:+.2f}σ)")
print(f"\n📡 Deploy this notebook on Zerve → live API endpoint in one click")

# Zerve reads this as the deployment output
api_response
