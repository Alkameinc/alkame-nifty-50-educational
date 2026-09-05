import { useState } from "react";
import SignalCard from "./components/SignalCard";
import "./App.css";

const SAMPLE_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"];

function App() {
  const [selectedSymbol, setSelectedSymbol] = useState(SAMPLE_SYMBOLS[0]);

  return (
    <div className="app-container">
      <header>
        <h1>Alkame-Nifty50</h1>
        <p className="tagline">
          Human-in-the-loop · Calibration-gated · Event-aware
        </p>
      </header>

      <label htmlFor="symbol-select">Stock:</label>

      <select
        id="symbol-select"
        value={selectedSymbol}
        onChange={(e) => setSelectedSymbol(e.target.value)}
      >
        {SAMPLE_SYMBOLS.map((sym) => (
          <option key={sym} value={sym}>
            {sym}
          </option>
        ))}
      </select>

      <SignalCard symbol={selectedSymbol} />
    </div>
  );
}

export default App;