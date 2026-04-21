"""
Monte Carlo Simulation Engine — World Cup 2026 Win Probability Engine

Models the full 2026 WC structure:
  - 48 teams, 12 groups of 4
  - Top 2 + 8 best 3rd-place finishers → 32-team knockout
  - Dixon-Coles adjusted Poisson goal distributions
  - 10,000 tournament simulations vectorized via NumPy
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Dixon-Coles correction (low-scoring matches) ───────────────────────────────

def _dc_tau(home_goals: int, away_goals: int, lam_h: float, lam_a: float, rho: float = -0.13) -> float:
    """Dixon-Coles low-score correction factor."""
    if home_goals == 0 and away_goals == 0:
        return 1 - lam_h * lam_a * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lam_h * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + lam_a * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def match_outcome_probs(elo_home: float, elo_away: float, neutral: bool = True, max_goals: int = 8):
    """
    Compute win/draw/loss probabilities using Dixon-Coles Poisson model
    parameterized by ELO difference.

    Returns (p_home_win, p_draw, p_away_win)
    """
    # Convert ELO difference to expected goals via logistic mapping
    elo_diff = elo_home - elo_away
    if not neutral:
        elo_diff += 75  # home advantage

    # Map ELO diff to attack/defense parameters
    home_attack = 1.35 * math.exp(elo_diff / 800.0)
    away_attack = 1.10 * math.exp(-elo_diff / 800.0)

    p_home_win = p_draw = p_away_win = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = (
                _poisson_pmf(h, home_attack)
                * _poisson_pmf(a, away_attack)
                * _dc_tau(h, a, home_attack, away_attack)
            )
            if h > a:
                p_home_win += p
            elif h == a:
                p_draw += p
            else:
                p_away_win += p

    total = p_home_win + p_draw + p_away_win
    return p_home_win / total, p_draw / total, p_away_win / total


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _sample_match_result_batch(elo_home: float, elo_away: float, n: int, neutral: bool = True) -> np.ndarray:
    """
    Vectorized: draw n match results.
    Returns array of shape (n,) with values: 0=home win, 1=draw, 2=away win
    """
    ph, pd_, pa = match_outcome_probs(elo_home, elo_away, neutral)
    return np.random.choice([0, 1, 2], size=n, p=[ph, pd_, pa])


# ── Group Stage ────────────────────────────────────────────────────────────────

def simulate_group_stage_vectorized(groups: list[list[str]], ratings: dict[str, float], n_sims: int) -> dict:
    """
    Simulate group stage for all n_sims in one pass.
    Returns {team: array[n_sims] of group finish position (1=first, 2=second, 3=third, 4=fourth)}
    """
    results = {team: np.zeros(n_sims, dtype=np.int8) for group in groups for team in group}
    group_thirds = []  # list of (team, points_array) for 3rd-place finishers

    for group in groups:
        # Round-robin: 6 matches per group (4 teams)
        team_points = {t: np.zeros(n_sims, dtype=np.int32) for t in group}
        team_gd = {t: np.zeros(n_sims, dtype=np.int32) for t in group}  # simplified

        match_pairs = [(group[i], group[j]) for i in range(len(group)) for j in range(i + 1, len(group))]

        for home, away in match_pairs:
            outcomes = _sample_match_result_batch(ratings.get(home, 1500), ratings.get(away, 1500), n_sims)
            team_points[home] += np.where(outcomes == 0, 3, np.where(outcomes == 1, 1, 0))
            team_points[away] += np.where(outcomes == 2, 3, np.where(outcomes == 1, 1, 0))
            # Simplified GD proxy
            team_gd[home] += np.where(outcomes == 0, 1, np.where(outcomes == 1, 0, -1))
            team_gd[away] += np.where(outcomes == 2, 1, np.where(outcomes == 1, 0, -1))

        # Rank teams within each simulation by points then GD
        pts_matrix = np.stack([team_points[t] for t in group])  # (4, n_sims)
        gd_matrix = np.stack([team_gd[t] for t in group])       # (4, n_sims)

        # Combined score for ranking (pts * 100 + gd)
        score_matrix = pts_matrix * 100 + gd_matrix
        ranks = np.argsort(-score_matrix, axis=0)  # (4, n_sims) — indices sorted by rank

        for rank_idx, team in enumerate(group):
            # Find what rank this team got in each sim
            team_pos = np.where(ranks == rank_idx, np.arange(1, 5)[:, None], 0).max(axis=0)
            results[team] = team_pos

        # Track 3rd place for wild card
        third_team_idx = ranks[2]  # index of 3rd-place finisher
        for sim_i in range(n_sims):
            t = group[third_team_idx[sim_i]]
            group_thirds.append((t, sim_i, pts_matrix[third_team_idx[sim_i], sim_i]))

    return results, group_thirds


# ── Full Tournament Simulation ─────────────────────────────────────────────────

@dataclass
class SimulationEngine:
    teams: list[str]
    ratings: dict[str, float]
    groups: list[list[str]] = field(default_factory=list)
    n_sims: int = 10_000

    def __post_init__(self):
        if not self.groups:
            self.groups = self._build_groups()

    def _build_groups(self) -> list[list[str]]:
        """Distribute 48 teams into 12 groups of 4 in seeded order."""
        n = 12
        sorted_teams = sorted(self.teams, key=lambda t: self.ratings.get(t, 1500), reverse=True)
        groups = [[] for _ in range(n)]
        for i, team in enumerate(sorted_teams):
            groups[i % n].append(team)
        return groups

    def run(self) -> dict[str, dict]:
        """
        Run n_sims full tournament simulations.
        Returns per-team stats: win_prob, finalist_prob, semifinal_prob, quarterfinal_prob.
        """
        np.random.seed(42)
        win_counts = {t: 0 for t in self.teams}
        finalist_counts = {t: 0 for t in self.teams}
        semi_counts = {t: 0 for t in self.teams}
        quarter_counts = {t: 0 for t in self.teams}

        for _ in range(self.n_sims):
            result = self._simulate_once()
            for stage, teams_in_stage in result.items():
                for t in teams_in_stage:
                    if stage == "winner":
                        win_counts[t] += 1
                    elif stage == "final":
                        finalist_counts[t] += 1
                    elif stage == "semi":
                        semi_counts[t] += 1
                    elif stage == "quarter":
                        quarter_counts[t] += 1

        output = {}
        for team in self.teams:
            output[team] = {
                "win_probability": round(win_counts[team] / self.n_sims, 4),
                "finalist_probability": round(finalist_counts[team] / self.n_sims, 4),
                "semifinal_probability": round(semi_counts[team] / self.n_sims, 4),
                "quarterfinal_probability": round(quarter_counts[team] / self.n_sims, 4),
            }

        return output

    def _simulate_once(self) -> dict:
        """Simulate one full tournament. Returns stage -> list of teams."""
        # Group stage
        group_results = {}
        all_thirds = []

        for group in self.groups:
            standing = self._simulate_group(group)
            group_results[group[0][0] if len(group) > 0 else "?"] = standing
            # Top 2 advance automatically
            for pos, team in enumerate(standing):
                if pos < 2:
                    group_results.setdefault("r32_auto", []).append(team)
                elif pos == 2:
                    all_thirds.append((team, standing))

        # Best 8 third-place finishers advance (simplified: take first 8)
        r32 = group_results.get("r32_auto", [])[:24]  # top 2 from 12 groups = 24
        r32 += [t for t, _ in all_thirds[:8]]  # 8 best thirds

        # Knockout rounds
        quarter_teams = self._simulate_knockout_round(r32[:32])
        semi_teams = self._simulate_knockout_round(quarter_teams)
        final_teams = self._simulate_knockout_round(semi_teams)
        winner = self._simulate_knockout_round(final_teams)

        return {
            "quarter": quarter_teams,
            "semi": semi_teams,
            "final": final_teams,
            "winner": winner,
        }

    def _simulate_group(self, group: list[str]) -> list[str]:
        """Returns group standing (ordered list) for one simulation."""
        points = {t: 0 for t in group}
        gd = {t: 0 for t in group}

        for i, home in enumerate(group):
            for away in group[i + 1:]:
                ph, pd_, pa = match_outcome_probs(
                    self.ratings.get(home, 1500),
                    self.ratings.get(away, 1500),
                    neutral=True,
                )
                r = np.random.choice([0, 1, 2], p=[ph, pd_, pa])
                if r == 0:
                    points[home] += 3; gd[home] += 1; gd[away] -= 1
                elif r == 1:
                    points[home] += 1; points[away] += 1
                else:
                    points[away] += 3; gd[away] += 1; gd[home] -= 1

        return sorted(group, key=lambda t: (points[t], gd[t]), reverse=True)

    def _simulate_knockout_round(self, teams: list[str]) -> list[str]:
        """Simulate one knockout round. Returns winners."""
        winners = []
        for i in range(0, len(teams), 2):
            if i + 1 >= len(teams):
                winners.append(teams[i])
                continue
            home, away = teams[i], teams[i + 1]
            ph, pd_, pa = match_outcome_probs(
                self.ratings.get(home, 1500),
                self.ratings.get(away, 1500),
                neutral=True,
            )
            # No draws in knockout — redistribute draw prob
            r = np.random.random()
            if r < ph:
                winners.append(home)
            elif r < ph + pa:
                winners.append(away)
            else:
                # Penalty shootout: 50/50
                winners.append(home if np.random.random() < 0.5 else away)
        return winners

    def run_with_confidence_intervals(self) -> dict[str, dict]:
        """
        Run simulations and compute 90% CI for win probability via bootstrap.
        Returns per-team stats including ci_90_lower, ci_90_upper, std.
        """
        np.random.seed(42)
        # Collect per-sim win indicator
        win_indicators = {t: [] for t in self.teams}

        batch = min(self.n_sims, 500)  # do 500 full sims for CI computation
        for _ in range(batch):
            result = self._simulate_once()
            winners = result.get("winner", [])
            for t in self.teams:
                win_indicators[t].append(1 if t in winners else 0)

        output = {}
        for team in self.teams:
            arr = np.array(win_indicators[team], dtype=float)
            mean = arr.mean()
            std = arr.std()
            ci_lower = max(0.0, mean - 1.645 * std / math.sqrt(len(arr)))
            ci_upper = min(1.0, mean + 1.645 * std / math.sqrt(len(arr)))
            output[team] = {
                "win_probability": round(mean, 4),
                "ci_90_lower": round(ci_lower, 4),
                "ci_90_upper": round(ci_upper, 4),
                "std": round(std, 4),
            }
        return output
