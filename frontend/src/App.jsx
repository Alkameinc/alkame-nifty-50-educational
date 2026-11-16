import { useState, useEffect } from "react";
import StockInsightsCard from "./components/StockInsightsCard";
import MarketBreadthPanel from "./components/MarketBreadthPanel";
import HealthBadge from "./components/HealthBadge";
import RiskBanner from "./components/RiskBanner";
import ErrorBoundary from "./components/ErrorBoundary";
import { fetchSymbols } from "./api";
import "./App.css";

function App() {
  const [symbols, setSymbols] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSymbols()
      .then((data) => {
        setSymbols(data.symbols || []);
        if (data.symbols && data.symbols.length > 0) {
          setSelectedSymbol(data.symbols[0]);
        }
      })
      .catch((err) => console.error("Failed to fetch symbols", err))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="app-container">Loading market intelligence cockpit...</div>;
  }

  return (
    <>
      {/* SEBI Compliance Banner */}
      <div style={{
        background: "rgba(245, 158, 11, 0.15)",
        borderBottom: "1px solid rgba(245, 158, 11, 0.35)",
        padding: "0.5rem 1rem",
        textAlign: "center",
        fontSize: "0.8rem",
        fontWeight: "bold",
        color: "#b45309"
      }}>
        ⚠️ SEBI REGULATORY NOTICE: Strictly educational and informational market analytics. No buy/sell recommendations, price targets, or trading advice provided.
      </div>

      <RiskBanner />

      <div className="app-container">
        <header>
          <h1>Alkame NIFTY 50</h1>
          <p className="tagline">
            Institutional Market Intelligence & Research Cockpit • Multi-Source Analytics
          </p>
          <HealthBadge />
        </header>

        <div className="controls" style={{ marginBottom: "1.5rem", display: "flex", gap: "1rem", justifyContent: "center", alignItems: "center" }}>
          <label htmlFor="symbol-select" style={{ fontWeight: "bold" }}>Select Constituent:</label>
          <select
            id="symbol-select"
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            style={{ padding: "0.5rem 1rem", fontSize: "1rem", borderRadius: "6px", border: "1px solid var(--border)", background: "var(--bg)", color: "inherit" }}
          >
            {symbols.map((sym) => (
              <option key={sym} value={sym}>
                {sym}
              </option>
            ))}
          </select>
        </div>

        <ErrorBoundary>
          {selectedSymbol && (
            <StockInsightsCard symbol={selectedSymbol} />
          )}
        </ErrorBoundary>

        <MarketBreadthPanel onSelectSymbol={setSelectedSymbol} />
      </div>
    </>
  );
}

export default App;