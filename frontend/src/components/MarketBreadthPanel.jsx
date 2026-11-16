import { useEffect, useState } from "react";
import { fetchMarketBreadth } from "../api";

export default function MarketBreadthPanel({ onSelectSymbol }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchMarketBreadth()
      .then((res) => setData(res))
      .catch((err) => console.error("Error fetching market breadth:", err))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div>Scanning NIFTY 50 market breadth...</div>;
  if (!data) return null;

  return (
    <div className="scalping-panel" style={{ marginTop: "2rem", padding: "1.25rem", border: "1px solid var(--border)", borderRadius: "8px", background: "var(--bg)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem", flexWrap: "wrap", gap: "0.5rem" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "1.25rem" }}>⚡ NIFTY 50 Volume & Momentum Radar</h2>
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Descriptive sweep across all 50 constituents for volume breakouts and momentum metrics
          </span>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", fontSize: "0.8rem", fontWeight: "bold" }}>
          <span style={{ color: "#16a34a", background: "rgba(22, 163, 74, 0.1)", padding: "0.25rem 0.5rem", borderRadius: "4px" }}>
            Advances: {data.advances}
          </span>
          <span style={{ color: "#dc2626", background: "rgba(220, 38, 38, 0.1)", padding: "0.25rem 0.5rem", borderRadius: "4px" }}>
            Declines: {data.declines}
          </span>
          <span style={{ color: "var(--text-muted)", background: "rgba(0,0,0,0.05)", padding: "0.25rem 0.5rem", borderRadius: "4px" }}>
            A/D Ratio: {data.advance_decline_ratio}
          </span>
        </div>
      </div>

      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", textAlign: "left", borderCollapse: "collapse", fontSize: "0.85rem" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid var(--border)", color: "var(--text-muted)" }}>
              <th style={{ padding: "8px" }}>Symbol</th>
              <th style={{ padding: "8px" }}>Sector</th>
              <th style={{ padding: "8px" }}>CMP</th>
              <th style={{ padding: "8px" }}>Day Change</th>
              <th style={{ padding: "8px" }}>Volume vs 20D</th>
              <th style={{ padding: "8px" }}>Surge Status</th>
              <th style={{ padding: "8px" }}>RSI (14)</th>
              <th style={{ padding: "8px" }}>52W Range</th>
              <th style={{ padding: "8px", textAlign: "right" }}>Inspect</th>
            </tr>
          </thead>
          <tbody>
            {(data.volume_surges && data.volume_surges.length > 0 ? data.volume_surges : data.top_gainers).map((item, idx) => (
              <tr key={idx} style={{ borderBottom: "1px solid var(--border)" }}>
                <td style={{ padding: "8px", fontWeight: "bold", color: "#2563eb" }}>{item.symbol}</td>
                <td style={{ padding: "8px", color: "var(--text-muted)" }}>{item.sector}</td>
                <td style={{ padding: "8px", fontWeight: "bold" }}>₹{item.cmp.toFixed(2)}</td>
                <td style={{ padding: "8px", fontWeight: "bold", color: item.day_change_pct >= 0 ? "#16a34a" : "#dc2626" }}>
                  {item.day_change_pct >= 0 ? `+${item.day_change_pct.toFixed(2)}%` : `${item.day_change_pct.toFixed(2)}%`}
                </td>
                <td style={{ padding: "8px", fontWeight: "bold" }}>{item.volume_ratio}x</td>
                <td style={{ padding: "8px" }}>
                  <span style={{
                    padding: "2px 6px",
                    borderRadius: "4px",
                    fontSize: "0.75rem",
                    fontWeight: "bold",
                    background: item.volume_surge ? "rgba(234, 88, 12, 0.15)" : "rgba(0,0,0,0.05)",
                    color: item.volume_surge ? "#ea580c" : "var(--text-muted)"
                  }}>
                    {item.volume_surge ? "🔥 SURGE" : "NORMAL"}
                  </span>
                </td>
                <td style={{ padding: "8px" }}>{item.rsi_14}</td>
                <td style={{ padding: "8px", color: "var(--text-muted)" }}>{item.pct_from_52w_high}% from 52H</td>
                <td style={{ padding: "8px", textAlign: "right" }}>
                  {onSelectSymbol && (
                    <button 
                      onClick={() => onSelectSymbol(item.symbol)}
                      style={{
                        padding: "4px 8px",
                        fontSize: "0.75rem",
                        fontWeight: "bold",
                        borderRadius: "4px",
                        border: "none",
                        background: "#2563eb",
                        color: "#fff",
                        cursor: "pointer"
                      }}
                    >
                      View
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
