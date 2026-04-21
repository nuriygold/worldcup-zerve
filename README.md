# World Cup 2026 Win Probability Engine
**ZerveHack Submission | April 2026**

> ELO ratings + Monte Carlo simulation + Polymarket divergence scoring.
> Find where prediction markets are wrong before the tournament starts.

---

## The Core Insight

Prediction markets systematically **overweight recent tournament performance** and **underweight long-run defensive consistency**. This engine quantifies that divergence in real time — team by team, normalized to simulation standard error — so you can see exactly which teams the market is mispricing.

---

## What It Does

- Ingests 20+ years of international A-match results (FBref / Sportsreference)
- Computes rolling ELO ratings with goal-margin scaling, home advantage, and exponential time decay (24-month half-life)
- Runs 10,000 Monte Carlo simulations of the full 2026 WC (48 teams, 12 groups, 32-team knockout)
- Uses Dixon-Coles adjusted Poisson distributions for realistic goal outcomes
- Cross-validates against live Polymarket implied probabilities (15-min cached)
- Scores divergence per team: `(P_model - P_market) / sigma_model`
- Flags teams where market odds fall outside the model 90% confidence interval

## Tech Stack

| Layer | Technology |
|---|---|
| Data pipeline | Python 3.11, pandas, NumPy |
| ELO + simulation | Custom engine (engine/) |
| API | FastAPI, async, rate-limited |
| Cache | Redis 7 (15-min TTL for odds) |
| Database | PostgreSQL 15 (ELO + sim results) |
| Frontend | React 18, Vite, Plotly.js, Zustand |
| Orchestration | Zerve |

---

## Quick Start

```bash
# Clone and configure
git clone <repo>
cp .env.example .env

# Docker (recommended)
docker-compose up

# Manual
cd backend && pip install -r requirements.txt
uvicorn api.main:app --reload &

cd frontend && npm install && npm run dev
```

API docs at `http://localhost:8000/docs`
Frontend at `http://localhost:5173`

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/probabilities` | GET | All 48 teams — win prob + CI + divergence |
| `/probabilities/{team}` | GET | Single team deep stats |
| `/divergence` | GET | Teams ranked by market gap |
| `/bracket/simulate` | POST | One full bracket simulation |
| `/odds/live` | GET | Polymarket odds snapshot |

---

## ELO Model Parameters

| Parameter | Value | Rationale |
|---|---|---|
| K factor (base) | 20 | Standard international |
| K factor (World Cup) | 40 | Higher tournament weight |
| Home advantage | +75 ELO | Neutral ground: 0 |
| Goal margin | log(diff+1) | Diminishing returns |
| Starting rating | 1500 | All new teams |
| Decay half-life | 24 months | exp(-λ·months), λ=ln(2)/24 |

---

## Known Limitations (Honest)

- Injury/suspension data not incorporated — a missing key player can shift true odds significantly
- Squad rotation for group stage not modeled
- Polymarket liquidity thin on lower-ranked teams (noisy implied probs)
- ELO doesn't yet account for opponent quality at time of match beyond the rating differential

---

## Project Structure

```
worldcup/
├── backend/
│   ├── api/main.py          # FastAPI app + all endpoints
│   ├── engine/
│   │   ├── elo.py           # ELO rating engine
│   │   ├── simulation.py    # Monte Carlo + Dixon-Coles
│   │   └── divergence.py    # Market divergence scoring
│   ├── data/
│   │   ├── fbref_loader.py  # FBref ingestion + cache
│   │   └── polymarket.py    # Odds fetcher + Redis cache
│   └── requirements.txt
├── frontend/
│   └── src/App.jsx          # Full React app (4 views)
├── docker-compose.yml
└── .env.example
```

---

Built by Nuriy / Adrian Cole (@nuriygold) for ZerveHack 2026
