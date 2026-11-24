import { useState, useEffect } from "react";
import SignalCard from "./components/SignalCard";
import "./App.css";

const DEFAULT_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "LT", "KOTAKBANK"];

function App() {
  const [symbols, setSymbols] = useState(DEFAULT_SYMBOLS);
  const [selectedSymbol, setSelectedSymbol] = useState(DEFAULT_SYMBOLS[0]);
  const [customSymbol, setCustomSymbol] = useState("");

  useEffect(() => {
    let cancelled = false;
    const API_BASE = (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_BASE_URL) || "http://127.0.0.1:8000";
    fetch(`${API_BASE}/symbols`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!cancelled && data && Array.isArray(data.symbols) && data.symbols.length > 0) {
          setSymbols(data.symbols);
        }
      })
      .catch(() => {
        // Fallback to DEFAULT_SYMBOLS on error
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleCustomSymbolSubmit = (e) => {
    e.preventDefault();
    const clean = customSymbol.trim().toUpperCase();
    if (clean) {
      if (!symbols.includes(clean)) {
        setSymbols((prev) => [...prev, clean]);
      }
      setSelectedSymbol(clean);
      setCustomSymbol("");
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Alkame-Nifty50</h1>
        <p className="tagline">
          Human-in-the-loop · Calibration-gated · Event-aware
        </p>
      </header>

      <div className="symbol-controls" style={{ marginBottom: "1rem", display: "flex", gap: "10px", alignItems: "center" }}>
        <label htmlFor="symbol-select">Stock:</label>
        <select
          id="symbol-select"
          value={selectedSymbol}
          onChange={(e) => setSelectedSymbol(e.target.value)}
        >
          {symbols.map((sym) => (
            <option key={sym} value={sym}>
              {sym}
            </option>
          ))}
        </select>

        <form onSubmit={handleCustomSymbolSubmit} style={{ display: "inline-flex", gap: "5px" }}>
          <input
            type="text"
            placeholder="Custom symbol (e.g. SBIN)"
            value={customSymbol}
            onChange={(e) => setCustomSymbol(e.target.value)}
            style={{ padding: "4px 8px" }}
          />
          <button type="submit" style={{ padding: "4px 8px" }}>Load Symbol</button>
        </form>
      </div>

      <SignalCard symbol={selectedSymbol} />
    </div>
  );
}

export default App;