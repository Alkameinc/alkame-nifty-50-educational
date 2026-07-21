import { useEffect, useState } from "react";
import { fetchSignal } from "../api";

function ConfidenceDisplay({ calibratedConfidence }) {
  // This mirrors predictor.py's own rule: never show a fake-precise number
  // until calibration has actually been proven. See Idea 1 in INTERN_ONBOARDING_GUIDE.md.
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

export default function SignalCard({ symbol }) {
  const [signal, setSignal] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchSignal(symbol)
      .then((data) => {
        if (!cancelled) setSignal(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true; // avoids setting state on an unmounted component if the user navigates away mid-request
    };
  }, [symbol]);

  if (loading) return <div className="signal-card">Loading signal for {symbol}...</div>;
  if (error) return <div className="signal-card signal-card-error">Error: {error}</div>;
  if (!signal) return null;

  return (
    <div className="signal-card">
      <div className="signal-card-header">
        <h2>{signal.symbol}</h2>
        <span className="action-badge" style={{ backgroundColor: actionColor(signal.action) }}>
          {signal.action}
        </span>
      </div>

      {signal.suppressed && (
        <p className="suppressed-notice">
          This signal was suppressed to HOLD. Reasons: {signal.suppression_reasons.join(" ")}
        </p>
      )}

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

      {signal.contributing_events.length > 0 && (
        <>
          <h3>Contributing events</h3>
          <ul>
            {signal.contributing_events.map((e) => (
              <li key={e.event_id}>
                [{e.scope}] {e.headline_or_label}
              </li>
            ))}
          </ul>
        </>
      )}

      <p className="global-risk-line">
        Global risk level: {signal.global_risk_level}
        {signal.risk_toggle_enabled ? " (risk toggle ENABLED)" : " (risk toggle off)"}
      </p>
    </div>
  );
}