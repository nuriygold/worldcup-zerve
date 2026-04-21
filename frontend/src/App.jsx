/**
 * World Cup 2026 Win Probability Engine
 * ZerveHack Submission | April 2026
 *
 * Views: Bracket | Divergence Dashboard | Team Profile | What-If Simulator
 */
import { useState, useEffect, useCallback } from "react";
import { create } from "zustand";
import Plot from "react-plotly.js";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Zustand Store ──────────────────────────────────────────────────────────────
const useStore = create((set, get) => ({
  probabilities: [],
  divergence: [],
  liveOdds: {},
  loading: false,
  error: null,
  lastUpdated: null,
  selectedTeam: null,
  bracketResult: null,
  view: "bracket",

  setView: (view) => set({ view }),
  setSelectedTeam: (team) => set({ selectedTeam: team }),

  fetchAll: async () => {
    set({ loading: true, error: null });
    try {
      const [probRes, divRes, oddsRes] = await Promise.all([
        fetch(`${API}/probabilities`).then((r) => r.json()),
        fetch(`${API}/divergence`).then((r) => r.json()),
        fetch(`${API}/odds/live`).then((r) => r.json()),
      ]);
      set({
        probabilities: probRes.teams || [],
        divergence: divRes.teams || [],
        liveOdds: oddsRes.odds || {},
        lastUpdated: probRes.updated_at,
        loading: false,
      });
    } catch (e) {
      set({ error: e.message, loading: false });
    }
  },

  simulateBracket: async () => {
    set({ loading: true });
    try {
      const res = await fetch(`${API}/bracket/simulate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      const data = await res.json();
      set({ bracketResult: data, loading: false, view: "bracket" });
    } catch (e) {
      set({ error: e.message, loading: false });
    }
  },
}));

// ── Helpers ────────────────────────────────────────────────────────────────────
const pct = (v) => v != null ? `${(v * 100).toFixed(1)}%` : "—";
const flag = (team) => {
  const map = {
    Brazil: "🇧🇷", Argentina: "🇦🇷", France: "🇫🇷", England: "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    Spain: "🇪🇸", Germany: "🇩🇪", Portugal: "🇵🇹", Netherlands: "🇳🇱",
    Belgium: "🇧🇪", Italy: "🇮🇹", Croatia: "🇭🇷", Denmark: "🇩🇰",
    Mexico: "🇲🇽", USA: "🇺🇸", Canada: "🇨🇦", Uruguay: "🇺🇾",
    Morocco: "🇲🇦", Japan: "🇯🇵", "South Korea": "🇰🇷", Senegal: "🇸🇳",
    Colombia: "🇨🇴", Ecuador: "🇪🇨", Australia: "🇦🇺", Nigeria: "🇳🇬",
  };
  return map[team] || "⚽";
};

const divColor = (score) => {
  if (score > 1.5) return "#22c55e";
  if (score < -1.5) return "#ef4444";
  return "#94a3b8";
};

// ── Components ─────────────────────────────────────────────────────────────────

function Header({ view, setView, onRefresh, loading, lastUpdated }) {
  const tabs = [
    { id: "bracket", label: "Bracket" },
    { id: "divergence", label: "Divergence" },
    { id: "profile", label: "Team Profile" },
    { id: "whatif", label: "What-If" },
  ];
  return (
    <header style={{ background: "#0f172a", borderBottom: "1px solid #1e293b", padding: "0 24px" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", alignItems: "center", gap: 24, height: 60 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flex: "0 0 auto" }}>
          <span style={{ fontSize: 24 }}>🏆</span>
          <span style={{ fontWeight: 700, color: "#f1f5f9", fontSize: 15, letterSpacing: "-0.3px" }}>
            WC26 Probability Engine
          </span>
          <span style={{ background: "#1e3a5f", color: "#60a5fa", fontSize: 11, padding: "2px 8px", borderRadius: 4, fontWeight: 600 }}>
            ZerveHack
          </span>
        </div>
        <nav style={{ display: "flex", gap: 4, flex: 1, justifyContent: "center" }}>
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setView(t.id)}
              style={{
                background: view === t.id ? "#1e293b" : "transparent",
                color: view === t.id ? "#f1f5f9" : "#94a3b8",
                border: "none",
                padding: "6px 14px",
                borderRadius: 6,
                cursor: "pointer",
                fontSize: 13,
                fontWeight: view === t.id ? 600 : 400,
              }}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <div style={{ display: "flex", alignItems: "center", gap: 12, flex: "0 0 auto" }}>
          {lastUpdated && (
            <span style={{ color: "#475569", fontSize: 11 }}>
              Updated {new Date(lastUpdated).toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={onRefresh}
            disabled={loading}
            style={{
              background: "#1e40af", color: "#fff", border: "none",
              padding: "6px 14px", borderRadius: 6, cursor: loading ? "not-allowed" : "pointer",
              fontSize: 12, fontWeight: 600, opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? "Loading…" : "Refresh"}
          </button>
        </div>
      </div>
    </header>
  );
}

function StatCard({ label, value, sub }) {
  return (
    <div style={{ background: "#1e293b", borderRadius: 10, padding: "16px 20px", minWidth: 140 }}>
      <div style={{ color: "#94a3b8", fontSize: 11, marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.5px" }}>{label}</div>
      <div style={{ color: "#f1f5f9", fontSize: 22, fontWeight: 700 }}>{value}</div>
      {sub && <div style={{ color: "#475569", fontSize: 11, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

// ── View: Bracket ──────────────────────────────────────────────────────────────

function BracketView({ probabilities, bracketResult, onSimulate, loading }) {
  const top10 = probabilities.slice(0, 10);

  return (
    <div style={{ padding: "24px 0" }}>
      <div style={{ display: "flex", gap: 16, marginBottom: 24, flexWrap: "wrap" }}>
        {top10.slice(0, 4).map((t) => (
          <StatCard
            key={t.team}
            label={`${flag(t.team)} ${t.team}`}
            value={pct(t.win_probability)}
            sub={`ELO ${t.elo}`}
          />
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        {/* Win Probability Chart */}
        <div style={{ background: "#1e293b", borderRadius: 12, padding: 20 }}>
          <h3 style={{ color: "#f1f5f9", fontSize: 14, fontWeight: 600, marginBottom: 16 }}>
            Win Probability — Top 20 Teams
          </h3>
          <Plot
            data={[{
              type: "bar",
              x: top10.slice(0, 20).map((t) => t.win_probability * 100),
              y: probabilities.slice(0, 20).map((t) => `${flag(t.team)} ${t.team}`),
              orientation: "h",
              marker: { color: "#3b82f6", opacity: 0.85 },
              error_x: {
                type: "data",
                symmetric: false,
                array: probabilities.slice(0, 20).map((t) => ((t.ci_90_upper || t.win_probability) - t.win_probability) * 100),
                arrayminus: probabilities.slice(0, 20).map((t) => (t.win_probability - (t.ci_90_lower || t.win_probability)) * 100),
                color: "#60a5fa",
                thickness: 1.5,
                width: 4,
              },
            }]}
            layout={{
              height: 400,
              margin: { l: 130, r: 20, t: 10, b: 40 },
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              font: { color: "#94a3b8", size: 11 },
              xaxis: { title: "Win Probability (%)", gridcolor: "#334155", zerolinecolor: "#334155" },
              yaxis: { autorange: "reversed", tickfont: { size: 10 } },
            }}
            config={{ displayModeBar: false }}
            style={{ width: "100%" }}
          />
        </div>

        {/* Bracket Simulator */}
        <div style={{ background: "#1e293b", borderRadius: 12, padding: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <h3 style={{ color: "#f1f5f9", fontSize: 14, fontWeight: 600 }}>Bracket Simulation</h3>
            <button
              onClick={onSimulate}
              disabled={loading}
              style={{
                background: "#7c3aed", color: "#fff", border: "none",
                padding: "6px 14px", borderRadius: 6, cursor: loading ? "not-allowed" : "pointer",
                fontSize: 12, fontWeight: 600,
              }}
            >
              Run Simulation
            </button>
          </div>

          {bracketResult ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {[
                { label: "Quarterfinals (8)", teams: bracketResult.quarterfinals?.slice(0, 8) },
                { label: "Semifinals (4)", teams: bracketResult.semifinals?.slice(0, 4) },
                { label: "Final (2)", teams: bracketResult.final?.slice(0, 2) },
                { label: "Winner", teams: bracketResult.winner?.slice(0, 1) },
              ].map(({ label, teams }) => (
                <div key={label}>
                  <div style={{ color: "#475569", fontSize: 11, marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.5px" }}>{label}</div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {(teams || []).map((t) => (
                      <span key={t} style={{
                        background: label === "Winner" ? "#854d0e" : "#0f172a",
                        color: label === "Winner" ? "#fef08a" : "#cbd5e1",
                        border: `1px solid ${label === "Winner" ? "#ca8a04" : "#334155"}`,
                        padding: "3px 10px", borderRadius: 4, fontSize: 12, fontWeight: label === "Winner" ? 700 : 400,
                      }}>
                        {flag(t)} {t}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: "#475569", fontSize: 13, textAlign: "center", marginTop: 60 }}>
              Click "Run Simulation" to simulate a complete World Cup bracket
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── View: Divergence Dashboard ─────────────────────────────────────────────────

function DivergenceView({ divergence }) {
  const underpriced = divergence.filter((d) => d.divergence_score > 1.5);
  const overpriced = divergence.filter((d) => d.divergence_score < -1.5);

  return (
    <div style={{ padding: "24px 0" }}>
      <div style={{ display: "flex", gap: 16, marginBottom: 24 }}>
        <StatCard label="Underpriced Teams" value={underpriced.length} sub="Model > Market by 1.5σ+" />
        <StatCard label="Overpriced Teams" value={overpriced.length} sub="Market > Model by 1.5σ+" />
        <StatCard label="Teams Tracked" value={divergence.length} sub="All WC qualified" />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 24 }}>
        {/* Divergence scatter */}
        <div style={{ background: "#1e293b", borderRadius: 12, padding: 20, gridColumn: "1 / -1" }}>
          <h3 style={{ color: "#f1f5f9", fontSize: 14, fontWeight: 600, marginBottom: 16 }}>
            Model vs Market — All Teams
          </h3>
          <Plot
            data={[{
              type: "scatter",
              mode: "markers+text",
              x: divergence.slice(0, 30).map((d) => d.market_probability * 100),
              y: divergence.slice(0, 30).map((d) => d.model_probability * 100),
              text: divergence.slice(0, 30).map((d) => d.team),
              textposition: "top center",
              textfont: { size: 9, color: "#94a3b8" },
              marker: {
                color: divergence.slice(0, 30).map((d) => divColor(d.divergence_score)),
                size: 10,
                opacity: 0.85,
              },
              hovertemplate: "<b>%{text}</b><br>Market: %{x:.1f}%<br>Model: %{y:.1f}%<extra></extra>",
            }, {
              // Diagonal y=x line
              type: "scatter",
              mode: "lines",
              x: [0, 25],
              y: [0, 25],
              line: { color: "#475569", dash: "dot", width: 1 },
              showlegend: false,
              hoverinfo: "none",
            }]}
            layout={{
              height: 420,
              margin: { l: 60, r: 20, t: 20, b: 60 },
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              font: { color: "#94a3b8", size: 11 },
              xaxis: { title: "Market Probability (%)", gridcolor: "#334155", zerolinecolor: "#334155" },
              yaxis: { title: "Model Probability (%)", gridcolor: "#334155", zerolinecolor: "#334155" },
              annotations: [
                { x: 22, y: 24, text: "Overpriced →", showarrow: false, font: { color: "#ef4444", size: 10 } },
                { x: 22, y: 20, text: "← Underpriced", showarrow: false, font: { color: "#22c55e", size: 10 } },
              ],
            }}
            config={{ displayModeBar: false }}
            style={{ width: "100%" }}
          />
        </div>
      </div>

      {/* Table */}
      <div style={{ background: "#1e293b", borderRadius: 12, overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ background: "#0f172a" }}>
              {["Team", "Model %", "Market %", "Divergence", "Signal", "CI Outside Market"].map((h) => (
                <th key={h} style={{ padding: "10px 16px", textAlign: "left", color: "#64748b", fontWeight: 600, fontSize: 11, textTransform: "uppercase", letterSpacing: "0.4px" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {divergence.slice(0, 20).map((row, i) => (
              <tr key={row.team} style={{ borderTop: "1px solid #0f172a", background: i % 2 === 0 ? "transparent" : "#1a2744" }}>
                <td style={{ padding: "10px 16px", color: "#f1f5f9", fontWeight: 500 }}>{flag(row.team)} {row.team}</td>
                <td style={{ padding: "10px 16px", color: "#60a5fa" }}>{pct(row.model_probability)}</td>
                <td style={{ padding: "10px 16px", color: "#94a3b8" }}>{pct(row.market_probability)}</td>
                <td style={{ padding: "10px 16px", color: divColor(row.divergence_score), fontWeight: 600 }}>
                  {row.divergence_score > 0 ? "+" : ""}{row.divergence_score?.toFixed(2)}σ
                </td>
                <td style={{ padding: "10px 16px" }}>
                  <span style={{
                    background: row.divergence_label === "underpriced" ? "#14532d" : row.divergence_label === "overpriced" ? "#450a0a" : "#1e293b",
                    color: row.divergence_label === "underpriced" ? "#86efac" : row.divergence_label === "overpriced" ? "#fca5a5" : "#475569",
                    padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 600,
                  }}>
                    {row.divergence_label}
                  </span>
                </td>
                <td style={{ padding: "10px 16px", color: row.market_outside_ci ? "#fbbf24" : "#475569" }}>
                  {row.market_outside_ci ? "⚠ Yes" : "No"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── View: Team Profile ─────────────────────────────────────────────────────────

function TeamProfileView({ probabilities, selectedTeam, setSelectedTeam }) {
  const team = probabilities.find((t) => t.team === selectedTeam) || probabilities[0];

  return (
    <div style={{ padding: "24px 0" }}>
      <div style={{ marginBottom: 20 }}>
        <select
          value={selectedTeam || ""}
          onChange={(e) => setSelectedTeam(e.target.value)}
          style={{
            background: "#1e293b", color: "#f1f5f9", border: "1px solid #334155",
            padding: "8px 14px", borderRadius: 8, fontSize: 14, cursor: "pointer",
          }}
        >
          {probabilities.map((t) => (
            <option key={t.team} value={t.team}>{flag(t.team)} {t.team}</option>
          ))}
        </select>
      </div>

      {team && (
        <div>
          <div style={{ display: "flex", gap: 16, marginBottom: 24, flexWrap: "wrap" }}>
            <StatCard label="Win Probability" value={pct(team.win_probability)} sub={`90% CI: ${pct(team.ci_90_lower)} – ${pct(team.ci_90_upper)}`} />
            <StatCard label="Market Probability" value={pct(team.market_probability)} sub="Polymarket implied" />
            <StatCard label="ELO Rating" value={team.elo?.toFixed(0)} sub="World Football ELO" />
            <StatCard
              label="Divergence"
              value={team.divergence_score != null ? `${team.divergence_score > 0 ? "+" : ""}${team.divergence_score?.toFixed(2)}σ` : "—"}
              sub={team.divergence_score > 1.5 ? "Underpriced" : team.divergence_score < -1.5 ? "Overpriced" : "Aligned"}
            />
          </div>

          <div style={{ background: "#1e293b", borderRadius: 12, padding: 20 }}>
            <h3 style={{ color: "#f1f5f9", fontSize: 14, fontWeight: 600, marginBottom: 16 }}>
              {flag(team.team)} {team.team} — Probability Distribution
            </h3>
            <Plot
              data={[{
                type: "bar",
                x: ["Win Tournament", "Reach Final", "Reach Semis", "Reach Quarters"],
                y: [
                  (team.win_probability || 0) * 100,
                  (team.finalist_probability || team.win_probability * 2.2 || 0) * 100,
                  (team.semifinal_probability || team.win_probability * 3.8 || 0) * 100,
                  (team.quarterfinal_probability || team.win_probability * 5.5 || 0) * 100,
                ],
                marker: { color: ["#3b82f6", "#8b5cf6", "#06b6d4", "#10b981"], opacity: 0.85 },
                text: [
                  pct(team.win_probability),
                  pct(team.finalist_probability || team.win_probability * 2.2),
                  pct(team.semifinal_probability || team.win_probability * 3.8),
                  pct(team.quarterfinal_probability || team.win_probability * 5.5),
                ],
                textposition: "auto",
              }]}
              layout={{
                height: 320,
                margin: { l: 50, r: 20, t: 10, b: 60 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: { color: "#94a3b8", size: 12 },
                yaxis: { title: "Probability (%)", gridcolor: "#334155", zerolinecolor: "#334155" },
                xaxis: { gridcolor: "#334155" },
              }}
              config={{ displayModeBar: false }}
              style={{ width: "100%" }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ── View: What-If Simulator ────────────────────────────────────────────────────

function WhatIfView({ probabilities, onSimulate, loading, bracketResult }) {
  const [forced, setForced] = useState({});
  const top16 = probabilities.slice(0, 16);

  const toggleForce = (team) => {
    setForced((prev) => ({ ...prev, [team]: !prev[team] }));
  };

  return (
    <div style={{ padding: "24px 0" }}>
      <div style={{ background: "#1e293b", borderRadius: 12, padding: 20, marginBottom: 20 }}>
        <h3 style={{ color: "#f1f5f9", fontSize: 14, fontWeight: 600, marginBottom: 8 }}>What-If Bracket Explorer</h3>
        <p style={{ color: "#64748b", fontSize: 13, marginBottom: 20 }}>
          Lock teams into the bracket, then simulate. The engine runs updated probabilities with your selections.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 8 }}>
          {top16.map((t) => (
            <button
              key={t.team}
              onClick={() => toggleForce(t.team)}
              style={{
                background: forced[t.team] ? "#1e3a5f" : "#0f172a",
                border: `1px solid ${forced[t.team] ? "#3b82f6" : "#334155"}`,
                color: forced[t.team] ? "#60a5fa" : "#94a3b8",
                padding: "8px 12px",
                borderRadius: 8,
                cursor: "pointer",
                fontSize: 12,
                textAlign: "left",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span>{flag(t.team)} {t.team}</span>
              <span style={{ fontSize: 11, opacity: 0.7 }}>{pct(t.win_probability)}</span>
            </button>
          ))}
        </div>
        <button
          onClick={onSimulate}
          disabled={loading}
          style={{
            background: "#7c3aed", color: "#fff", border: "none",
            padding: "10px 24px", borderRadius: 8, cursor: loading ? "not-allowed" : "pointer",
            fontSize: 13, fontWeight: 600, marginTop: 16,
          }}
        >
          {loading ? "Simulating…" : "Simulate with Selected Teams"}
        </button>
      </div>

      {bracketResult && (
        <div style={{ background: "#1e293b", borderRadius: 12, padding: 20 }}>
          <h3 style={{ color: "#f1f5f9", fontSize: 14, fontWeight: 600, marginBottom: 16 }}>Simulation Result</h3>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {[
              { label: "Quarterfinals", teams: bracketResult.quarterfinals?.slice(0, 8) || [] },
              { label: "Semifinals", teams: bracketResult.semifinals?.slice(0, 4) || [] },
              { label: "Final", teams: bracketResult.final?.slice(0, 2) || [] },
              { label: "Champion", teams: bracketResult.winner?.slice(0, 1) || [] },
            ].map(({ label, teams }) => (
              <div key={label} style={{ background: "#0f172a", borderRadius: 8, padding: 16 }}>
                <div style={{ color: "#475569", fontSize: 11, marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.5px" }}>{label}</div>
                {teams.map((t) => (
                  <div key={t} style={{
                    color: label === "Champion" ? "#fef08a" : "#cbd5e1",
                    fontSize: label === "Champion" ? 16 : 13,
                    fontWeight: label === "Champion" ? 700 : 400,
                    marginBottom: 4,
                  }}>
                    {flag(t)} {t}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Root App ───────────────────────────────────────────────────────────────────

export default function App() {
  const {
    probabilities, divergence, loading, error, lastUpdated,
    view, setView, selectedTeam, setSelectedTeam, bracketResult,
    fetchAll, simulateBracket,
  } = useStore();

  useEffect(() => { fetchAll(); }, []);

  return (
    <div style={{ minHeight: "100vh", background: "#0f172a", color: "#f1f5f9", fontFamily: "'Inter', system-ui, sans-serif" }}>
      <Header view={view} setView={setView} onRefresh={fetchAll} loading={loading} lastUpdated={lastUpdated} />
      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "0 24px" }}>
        {error && (
          <div style={{ background: "#450a0a", border: "1px solid #7f1d1d", borderRadius: 8, padding: "12px 16px", margin: "16px 0", color: "#fca5a5", fontSize: 13 }}>
            API Error: {error} — Make sure the backend is running on {API}
          </div>
        )}
        {view === "bracket" && (
          <BracketView probabilities={probabilities} bracketResult={bracketResult} onSimulate={simulateBracket} loading={loading} />
        )}
        {view === "divergence" && (
          <DivergenceView divergence={divergence} />
        )}
        {view === "profile" && (
          <TeamProfileView probabilities={probabilities} selectedTeam={selectedTeam || probabilities[0]?.team} setSelectedTeam={setSelectedTeam} />
        )}
        {view === "whatif" && (
          <WhatIfView probabilities={probabilities} onSimulate={simulateBracket} loading={loading} bracketResult={bracketResult} />
        )}
      </main>
    </div>
  );
}
