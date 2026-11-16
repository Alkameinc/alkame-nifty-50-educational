import { useEffect, useState } from "react";
import { fetchInsights } from "../api";

export default function StockInsightsCard({ symbol }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchInsights(symbol)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [symbol]);

  if (loading) {
    return <div className="signal-card">Loading market intelligence for {symbol}...</div>;
  }

  if (error) {
    return <div className="signal-card signal-card-error">Error loading insights: {error}</div>;
  }

  if (!data) return null;

  const { technical, valuation, corporate_events, news_sentiment, macro_context } = data;

  return (
    <div className="signal-card" style={{ borderTop: "4px solid var(--accent, #2563eb)" }}>
      {/* Header */}
      <div className="signal-card-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "1.6rem" }}>{data.symbol}</h2>
          <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
            {valuation.sector || "NIFTY 50 Constituent"} • {valuation.industry || "General"}
          </span>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: "1.6rem", fontWeight: "bold" }}>₹{data.cmp.toFixed(2)}</div>
          <span style={{ 
            fontSize: "0.85rem", 
            fontWeight: "bold",
            color: data.day_change_pct >= 0 ? "#16a34a" : "#dc2626" 
          }}>
            {data.day_change_pct >= 0 ? `+${data.day_change_pct.toFixed(2)}%` : `${data.day_change_pct.toFixed(2)}%`}
          </span>
        </div>
      </div>

      {/* Statutory Disclaimer Tag */}
      <div style={{
        marginTop: "0.75rem",
        padding: "0.5rem 0.75rem",
        background: "rgba(245, 158, 11, 0.1)",
        border: "1px solid rgba(245, 158, 11, 0.3)",
        borderRadius: "6px",
        fontSize: "0.75rem",
        color: "#b45309"
      }}>
        🛡️ <strong>Educational Analytics:</strong> Pure descriptive mathematics and factual data aggregation. No buy/sell recommendations, price targets, or stop-loss calls provided.
      </div>

      {/* Fundamental Valuation Grid */}
      <h3 style={{ marginTop: "1.25rem", marginBottom: "0.5rem", fontSize: "1rem" }}>📊 Fundamental Valuation & Profile</h3>
      <div style={{ 
        display: "grid", 
        gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", 
        gap: "0.75rem",
        background: "var(--card-bg, rgba(0,0,0,0.02))",
        padding: "0.75rem",
        borderRadius: "8px",
        border: "1px solid var(--border)"
      }}>
        <div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>P/E Ratio</div>
          <div style={{ fontWeight: "bold", fontSize: "1rem" }}>{valuation.pe_ratio || "N/A"}</div>
        </div>
        <div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>P/B Ratio</div>
          <div style={{ fontWeight: "bold", fontSize: "1rem" }}>{valuation.price_to_book || "N/A"}</div>
        </div>
        <div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Market Cap</div>
          <div style={{ fontWeight: "bold", fontSize: "1rem" }}>{valuation.market_cap || "N/A"}</div>
        </div>
        <div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>52W High / Low</div>
          <div style={{ fontWeight: "bold", fontSize: "0.9rem" }}>₹{technical.fifty_two_week_high || "N/A"} / ₹{technical.fifty_two_week_low || "N/A"}</div>
        </div>
        <div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Trailing EPS</div>
          <div style={{ fontWeight: "bold", fontSize: "1rem" }}>₹{valuation.eps || "N/A"}</div>
        </div>
        <div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Dividend Yield</div>
          <div style={{ fontWeight: "bold", fontSize: "1rem" }}>{valuation.dividend_yield || "N/A"}</div>
        </div>
      </div>

      {/* Technical Posture Grid */}
      <h3 style={{ marginTop: "1.25rem", marginBottom: "0.5rem", fontSize: "1rem" }}>🧭 Technical Indicators & Reference Levels</h3>
      <div style={{ 
        display: "grid", 
        gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", 
        gap: "0.75rem",
        background: "var(--card-bg, rgba(0,0,0,0.02))",
        padding: "0.75rem",
        borderRadius: "8px",
        border: "1px solid var(--border)"
      }}>
        <div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>RSI (14-Day)</div>
          <div style={{ fontWeight: "bold", fontSize: "1.1rem", color: technical.rsi_14 >= 70 ? "#ea580c" : technical.rsi_14 <= 30 ? "#2563eb" : "inherit" }}>
            {technical.rsi_14}
          </div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>{technical.rsi_status}</div>
        </div>

        <div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Volume vs 20D SMA</div>
          <div style={{ fontWeight: "bold", fontSize: "1.1rem" }}>
            {technical.volume_ratio ? `${technical.volume_ratio}x` : "1.0x"}
          </div>
          <div style={{ fontSize: "0.7rem", color: technical.volume_surge ? "#ea580c" : "var(--text-muted)", fontWeight: technical.volume_surge ? "bold" : "normal" }}>
            {technical.volume_surge ? "🔥 Volume Surge" : "Normal Flow"}
          </div>
        </div>

        <div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Trend Posture (EMAs)</div>
          <div style={{ fontWeight: "bold", fontSize: "0.9rem" }}>{technical.trend_posture}</div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>EMA20: ₹{technical.ema_20 || "N/A"}</div>
        </div>

        <div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>VWAP Anchor</div>
          <div style={{ fontWeight: "bold", fontSize: "1.1rem" }}>{technical.vwap ? `₹${technical.vwap}` : "N/A"}</div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>{technical.vwap_relation}</div>
        </div>

        <div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Bollinger %B / Bandwidth</div>
          <div style={{ fontWeight: "bold", fontSize: "0.95rem" }}>
            %B: {technical.bollinger_pct_b ?? "N/A"}
          </div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>BW: {technical.bollinger_bandwidth ?? "N/A"}</div>
        </div>

        <div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Camarilla S/R Levels</div>
          <div style={{ fontSize: "0.75rem", fontWeight: "bold" }}>H3: ₹{technical.pivots_camarilla?.H3} | L3: ₹{technical.pivots_camarilla?.L3}</div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>H4: ₹{technical.pivots_camarilla?.H4} | L4: ₹{technical.pivots_camarilla?.L4}</div>
        </div>
      </div>

      {/* Corporate Events Feed */}
      {corporate_events && corporate_events.length > 0 && (
        <div style={{ marginTop: "1.25rem" }}>
          <h3 style={{ marginBottom: "0.5rem", fontSize: "1rem" }}>📡 Corporate Announcements & Filings (NSE)</h3>
          <ul style={{ paddingLeft: "1.2rem", margin: 0, fontSize: "0.85rem", color: "var(--text-main)" }}>
            {corporate_events.map((ev, i) => (
              <li key={i} style={{ marginBottom: "0.4rem" }}>
                <strong>[{ev.category}]</strong> {ev.headline} <span style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>({ev.date})</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* News Sentiment Section */}
      {news_sentiment && (
        <div style={{ marginTop: "1.25rem" }}>
          <h3 style={{ marginBottom: "0.3rem", fontSize: "1rem" }}>📰 Public News Sentiment & Polarity</h3>
          <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.5rem" }}>
            Overall Polarity: <strong>{news_sentiment.sentiment_label}</strong> (Score: {news_sentiment.sentiment_score}) across {news_sentiment.articles_analyzed} analyzed news items.
          </div>
          {news_sentiment.recent_headlines && news_sentiment.recent_headlines.length > 0 && (
            <ul style={{ paddingLeft: "1.2rem", margin: 0, fontSize: "0.8rem" }}>
              {news_sentiment.recent_headlines.map((item, idx) => (
                <li key={idx} style={{ marginBottom: "0.3rem" }}>
                  {item.title} <span style={{ color: "var(--text-muted)" }}>— {item.source}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
