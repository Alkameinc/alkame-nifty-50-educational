import { useState, useEffect } from "react";
import SignalCard from "./components/SignalCard";
import HorizonTabs from "./components/HorizonTabs";
import HealthBadge from "./components/HealthBadge";
import ScalpingPanel from "./components/ScalpingPanel";
import RiskBanner from "./components/RiskBanner";
import ErrorBoundary from "./components/ErrorBoundary";
import { fetchSymbols } from "./api";
import "./App.css";

function App() {
  const [symbols, setSymbols] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState("");
  const [selectedHorizon, setSelectedHorizon] = useState("INTRADAY");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSymbols()
      .then(data => {
        setSymbols(data.symbols);
        if (data.symbols.length > 0) {
          setSelectedSymbol(data.symbols[0]);
        }
      })
      .catch(err => console.error("Failed to fetch symbols", err))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="app-container">Loading app...</div>;
  }

  return (
    <>
      <RiskBanner />
      <div className="app-container">
        <header>
          <h1>Alkame-Nifty50</h1>
          <p className="tagline">
            Human-in-the-loop · Calibration-gated · Event-aware
          </p>
          <HealthBadge />
        </header>

        <div className="controls" style={{ marginBottom: "2rem", display: "flex", gap: "1rem", justifyContent: "center", alignItems: "center" }}>
          <label htmlFor="symbol-select" style={{ fontWeight: 'bold' }}>Select Stock:</label>
          <select
            id="symbol-select"
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            style={{ padding: '0.5rem', fontSize: '1rem', borderRadius: '4px', border: '1px solid var(--border)' }}
          >
            {symbols.map((sym) => (
              <option key={sym} value={sym}>
                {sym}
              </option>
            ))}
          </select>
        </div>

        <HorizonTabs 
          horizons={["INTRADAY", "3D", "7D", "30D", "3M", "6M", "1Y"]}
          selected={selectedHorizon}
          onSelect={setSelectedHorizon}
        />

        <ErrorBoundary>
          {selectedSymbol && (
            <SignalCard symbol={selectedSymbol} horizon={selectedHorizon} />
          )}
        </ErrorBoundary>
        
        {selectedHorizon === "INTRADAY" && <ScalpingPanel />}
      </div>
    </>
  );
}

export default App;