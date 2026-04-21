# ⚽ World Cup 2026 Win Probability Engine

> **ZerveHack 2026 Submission** | $10,000 in prizes | Deadline April 29 @ 2pm EDT

🚀 **Live Demo:** [https://73565948-87cc4798.hub.zerve.cloud](https://73565948-87cc4798.hub.zerve.cloud) · [Zerve Notebook](https://app.zerve.ai/notebook/a35a42cf-ade7-4ca0-86b5-cb4648662dd7)

[![Built with Zerve](https://img.shields.io/badge/Built%20with-Zerve-6C47FF?style=flat-square)](https://zerve.ai)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

---

## 🧠 The Core Insight

Prediction markets **systematically overweight recent tournament performance** and **underweight long-run defensive consistency.**

This engine quantifies that gap — in real time, team by team — so you can see exactly where Polymarket is wrong before a single ball is kicked.

```
divergence = (P_model − P_market) / σ_model

+1.5σ → market underpricing this team  🟢
−1.5σ → market overpricing this team   🔴
```

---

## 🟣 What Zerve Does in This App

Zerve isn't just listed in the stack — it **is** the infrastructure. Here's exactly what it handles:

| Zerve Capability | How We Use It |
|---|---|
| **Collaborative notebooks** | Built and iterated the ELO engine and Dixon-Coles simulation in Zerve's Python notebooks — isolated compute per block, no kernel crashes mid-10k sim |
| **AI agent** | Zerve's agent caught edge cases in the Monte Carlo group-stage tiebreaker logic and iterated on the Dixon-Coles low-score correction (0-0, 1-0, 1-1 fixes) |
| **One-click API deployment** | The FastAPI layer (`/probabilities`, `/divergence`, `/bracket/simulate`) is deployed and live via Zerve — no servers, no YAML, no DevOps handoff |
| **Scheduled pipeline execution** | Zerve runs the nightly FBref cron (02:00 UTC), triggers ELO recalculation on new match data, and re-runs the full 10k simulation batch every 6 hours during the tournament window |
| **Orchestration** | Zerve coordinates the Polymarket odds refresh loop (15-min TTL via Redis) and fires divergence rescoring on every odds update |

Without Zerve: notebooks → hand off to engineers → wait → deploy → break → repeat.

With Zerve: **notebooks → deploy → live.** Same session.

---

## 📐 Architecture

```
FBref (20yr match data)          Polymarket API
         │                              │
         ▼                              ▼
  pandas pipeline              Redis cache (15-min TTL)
  ELO computation  ─────────►  Divergence scoring
         │                              │
         ▼                              │
  Monte Carlo engine                   │
  10,000 tournament sims               │
  Dixon-Coles Poisson model            │
         │                             │
         └─────────────┬───────────────┘
                       ▼
              ┌──────────────────┐
              │  FastAPI REST    │
              │  /probabilities  │
              │  /divergence     │
              │  /bracket/sim    │
              │  /odds/live      │
              └──────────────────┘
                       │
              ┌─────────────────────────┐
              │     React 18 UI         │
              │  Bracket View           │
              │  Divergence Dashboard   │
              │  Team Profile           │
              │  What-If Simulator      │
              └─────────────────────────┘
```

All pipeline scheduling and API serving runs through **Zerve**.

---

## ⚙️ ELO Model

Built from scratch on 20+ years of international A-match results (approx. 28,000 matches).

| Parameter | Value | Why |
|---|---|---|
| K factor (base) | 20 | Standard international calibration |
| K factor (World Cup) | 40 | Tournament results carry more signal |
| Home advantage | +75 ELO | Zero on neutral ground |
| Goal margin | `log(diff + 1)` | Diminishing returns on blowouts |
| Starting rating | 1500 | All new teams equal |
| Decay half-life | 24 months | `exp(−λ·months)`, λ = ln(2)/24 |

The 2006 World Cup does not carry equal weight to a 2024 Nations League result. The model knows this.

---

## 🎲 Monte Carlo Simulation

**48 teams. 12 groups. 32-team knockout. 10,000 runs.**

Each simulated match draws from a **Dixon-Coles adjusted Poisson distribution** — producing a full joint scoreline distribution, not a binary win/loss.

```python
λ_home = exp(α_home + β_away + γ)   # attack strength − defensive weakness + home factor
λ_away = exp(α_away + β_home)

# Dixon-Coles correction on low-scoring outcomes prevents
# systematic underestimation of 0-0, 1-0, 0-1, 1-1 scorelines
```

Full 10,000-sim batch completes in ~4 seconds via NumPy vectorization.

---

## 📡 REST API

| Endpoint | Method | Description |
|---|---|---|
| `/probabilities` | `GET` | All 48 teams — win prob + 90% CI + ELO + divergence score |
| `/probabilities/{team}` | `GET` | Single team deep stats |
| `/divergence` | `GET` | Teams ranked by market gap (absolute σ), descending |
| `/bracket/simulate` | `POST` | One full stochastic bracket — group stage through final |
| `/odds/live` | `GET` | Current Polymarket snapshot (15-min Redis cache) |

Rate limited at 60 req/min per IP. No auth required — public data, public model.

---

## 🖥️ Frontend — 4 Views

- 🏆 **Bracket** — Top 20 win probabilities with 90% CI error bars + one-click bracket simulation
- 📊 **Divergence Dashboard** — Model vs. market scatter, color-coded by signal direction, full ranked table
- 🌍 **Team Profile** — ELO history, win/finalist/semi/quarter probabilities per team
- 🎯 **What-If Simulator** — Lock teams into the bracket, run alternate timelines, see how probabilities shift

Stack: React 18 · Vite · Plotly.js · Zustand

---

## 🛠️ Quick Start

```bash
git clone https://github.com/nuriygold/worldcup-zerve.git
cd worldcup-zerve
cp .env.example .env

# One command — Redis + API + Frontend
docker-compose up
```

- API docs → `http://localhost:8000/docs`
- Frontend → `http://localhost:5173`

---

## 📁 Project Structure

```
worldcup-zerve/
├── backend/
│   ├── api/main.py           # FastAPI — all 5 endpoints, rate limiting, CORS
│   ├── engine/
│   │   ├── elo.py            # ELO rating engine with time decay
│   │   ├── simulation.py     # Monte Carlo + Dixon-Coles Poisson model
│   │   └── divergence.py     # Market divergence scoring
│   └── data/
│       ├── fbref_loader.py   # FBref ingestion + synthetic fallback
│       └── polymarket.py     # Odds fetch + Redis cache (15-min TTL)
├── frontend/src/App.jsx      # Full React app — 4 views, Plotly charts
├── docker-compose.yml
└── .env.example
```

---

## ⚠️ Known Limitations

- **Injuries not modeled** — a missing Mbappé can shift true odds more than any rating system can see
- **Squad rotation ignored** — coaches resting starters in group stage changes real match outcomes
- **Polymarket liquidity** — thin on lower-ranked teams, implied probs can be noisy
- **ELO opponent weighting** — doesn't yet adjust for opponent strength at time of match beyond raw rating differential

---

## 🏆 ZerveHack 2026

**Team:** Nuriy Gold | [@nuriygold](https://github.com/nuriygold) | hello@nuriy.com

**Theme:** Sports Analytics — find the market inefficiency before the tournament starts.

**Prize pool:** $5,000 · $3,000 · $2,000 | Deadline: April 29, 2026 @ 2:00pm EDT

---

*Built on [Zerve](https://zerve.ai) — the AI-native data science platform where notebooks deploy as live APIs.*
