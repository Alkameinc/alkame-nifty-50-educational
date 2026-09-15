import { useEffect, useState } from "react";
import { fetchSignal } from "../api";
import MiniChart from "./MiniChart";
import NarrativeSummary from "./NarrativeSummary";

function ConfidenceDisplay({ calibratedConfidence }) {
  if (calibratedConfidence === null || calibratedConfidence === undefined) {
    return (
      <p className="confidence-warning">
        ⚠️ Confidence not yet calibrated — not enough resolved history for this symbol yet.
        Treat this signal as directional only, not a trustworthy probability.
      </p>
    );
  }
  return <p className="confidence-value">Calibrated confidence: {(calibratedConfidence * 100).toFixed(1)}%</p>;
}

function actionColor(action) {
  if (action === "BUY") return "#1a7f37";
  if (action === "SELL") return "#cf222e";
  return "#57606a"; // HOLD
}

export default function SignalCard({ symbol, horizon }) {
  const [fullData, setFullData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    const loadData = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchSignal(symbol);
        if (!cancelled) setFullData(data);
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    loadData();

    return () => {
      cancelled = true;
    };
  }, [symbol]);

  if (loading) return <div className="signal-card">Loading signal for {symbol}...</div>;
  if (error) return <div className="signal-card signal-card-error">Error: {error}</div>;
  if (!fullData || !fullData.signals) return null;

  const signal = fullData.signals[horizon];
  if (!signal) {
     return <div className="signal-card">No signal data available for horizon: {horizon}</div>;
  }

  return (
    <div className="signal-card">
      <div className="signal-card-header">
        <h2>{fullData.symbol} ({horizon})</h2>
        <span className="action-badge" style={{ backgroundColor: actionColor(signal.action) }}>
          {signal.action}
        </span>
      </div>

      <div className="price-info" style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '1rem', fontSize: '0.9rem' }}>
         {signal.current_price && <div><strong>Current:</strong> ₹{signal.current_price.toFixed(2)}</div>}
         {signal.target_price && <div><strong>Target:</strong> ₹{signal.target_price.toFixed(2)}</div>}
         {signal.stop_loss && <div><strong>Stop Loss:</strong> ₹{signal.stop_loss.toFixed(2)}</div>}
      </div>

      <MiniChart symbol={fullData.symbol} horizon={horizon} />

      <NarrativeSummary narrative={fullData.narrative} />

      <p className="verdict-text" style={{ fontWeight: 'bold', marginBottom: '1rem' }}>
        {signal.verdict_text}
      </p>

      <ConfidenceDisplay calibratedConfidence={signal.calibrated_confidence} />

      <h3>Downside (read this first)</h3>
      <p>{signal.downside_summary}</p>

      <h3>Upside</h3>
      <p>{signal.upside_summary}</p>

      <h3>Reasoning</h3>
      <ul>
        {signal.reasoning.map((line, i) => (
          <li key={i}>{line}</li>
        ))}
      </ul>

      {signal.events && signal.events.length > 0 && (
        <>
          <h3>Contributing events</h3>
          <ul>
            {signal.events.map((e, i) => (
              <li key={i}>
                [{e.type}] {e.label} {e.sentiment ? `(Score: ${e.sentiment})` : ""}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}