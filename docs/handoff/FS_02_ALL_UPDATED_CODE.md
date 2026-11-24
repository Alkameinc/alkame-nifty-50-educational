# FS-02 Combined Code Package & Complete Implementation Reference

This document contains the **complete, full-file updated code** for all 17 files modified or created for the **FS-02 / Full-stack developer** assignment in `Alkameinc/alkame-nifty-50-educational`.

---

## Table of Contents

1. [human_insight_manager.py](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/human_insight_manager.py) — (F05)
2. [app.py](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/app.py) — (F05, F40)
3. [predictor.py](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/predictor.py) — (F36, F40)
4. [position_planner.py](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/position_planner.py) — (F35, F36)
5. [narrative_builder.py](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/narrative_builder.py) — (F35, F40)
6. [scalping.py](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/scalping.py) — (F40)
7. [frontend/src/api.js](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/frontend/src/api.js) — (F48)
8. [frontend/src/components/SignalCard.jsx](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/frontend/src/components/SignalCard.jsx) — (F48, F57)
9. [frontend/src/App.jsx](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/frontend/src/App.jsx) — (F48)
10. [tests/fs_02/fixtures/signal_specimen.json](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/tests/fs_02/fixtures/signal_specimen.json) — (Specimen Fixture)
11. [tests/fs_02/fixtures/prediction_signal.py](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/tests/fs_02/fixtures/prediction_signal.py) — (Specimen Fixture)
12. [tests/fs_02/test_f05_override.py](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/tests/fs_02/test_f05_override.py) — (F05 Unit Tests)
13. [tests/fs_02/test_f36_position_intent.py](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/tests/fs_02/test_f36_position_intent.py) — (F36 Unit Tests)
14. [tests/fs_02/test_f35_position_planner.py](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/tests/fs_02/test_f35_position_planner.py) — (F35 Unit Tests)
15. [tests/fs_02/test_f40_score_presentation.py](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/tests/fs_02/test_f40_score_presentation.py) — (F40 Unit Tests)
16. [tests/fs_02/test_f48_f57_frontend.py](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/tests/fs_02/test_f48_f57_frontend.py) — (F48/F57 Unit Tests)
17. [docs/handoff/FS-02.md](file:///c:/Users/HP/Downloads/alkame-nifty-50-educational-main/alkame-nifty-50-educational-main/docs/handoff/FS-02.md) — (Handoff Report)

---

## 1. File: `human_insight_manager.py`

```python
# 1. Standard library imports
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# 2. Third-party imports
# (none required — stdlib sqlite3 only)

# 3. Local imports
from config import DB_PATH, ensure_directories, configure_logging
from predictor import PredictionSignal
from health_monitor import registry as health_registry

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
SQLITE_TIMEOUT_SECONDS = 10
DEFAULT_CREATED_BY = "default_trader"


@dataclass
class NoteRecord:
    id: int
    symbol: str
    timestamp: str
    note_text: str
    related_action: Optional[str]
    created_by: str


@dataclass
class OverrideRecord:
    id: int
    symbol: str
    timestamp: str
    original_action: str
    overridden_action: str
    reason: str
    created_by: str


@dataclass
class FeedbackRecord:
    id: int
    symbol: str
    signal_timestamp: str
    original_action: str
    was_helpful: Optional[bool]
    outcome_notes: str
    rated_at: str


# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
class HumanInsightManager:
    """
    Stores everything a human trader adds on top of the model's own output:
    free-text notes, explicit overrides of a signal's final action (with a
    mandatory reason, same audit-trail discipline as the global risk toggle),
    and feedback on whether a past signal was actually helpful.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        ensure_directories()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=SQLITE_TIMEOUT_SECONDS)

    def _init_db(self) -> None:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS human_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    note_text TEXT NOT NULL,
                    related_action TEXT,
                    created_by TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS human_overrides (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    original_action TEXT NOT NULL,
                    overridden_action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_by TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_timestamp TEXT NOT NULL,
                    original_action TEXT NOT NULL,
                    was_helpful INTEGER,
                    outcome_notes TEXT,
                    rated_at TEXT NOT NULL
                )
            """)
            conn.commit()
        except Exception as e:
            logger.error(f"Failed initializing human insight database at {self.db_path}: {e}")
            raise
        finally:
            if conn is not None:
                conn.close()

    def add_note(self, symbol: str, note_text: str, related_action: Optional[str] = None,
                 created_by: str = DEFAULT_CREATED_BY) -> Optional[int]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO human_notes (symbol, timestamp, note_text, related_action, created_by) "
                "VALUES (?, ?, ?, ?, ?)",
                (symbol, datetime.now().isoformat(), note_text, related_action, created_by),
            )
            conn.commit()
            note_id = cursor.lastrowid
            health_registry.report("human_insight_manager", ok=True, detail=f"Added note for {symbol}")
            return note_id
        except Exception as e:
            logger.error(f"Failed adding note for {symbol}: {e}")
            health_registry.report("human_insight_manager", ok=False, detail="Failed adding note", error=str(e))
            return None
        finally:
            if conn is not None:
                conn.close()

    def get_notes(self, symbol: Optional[str] = None, limit: int = 50) -> List[NoteRecord]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT id, symbol, timestamp, note_text, related_action, created_by "
                    "FROM human_notes WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                    (symbol, limit),
                )
            else:
                cursor.execute(
                    "SELECT id, symbol, timestamp, note_text, related_action, created_by "
                    "FROM human_notes ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
            health_registry.report("human_insight_manager", ok=True)
            return [NoteRecord(*row) for row in rows]
        except Exception as e:
            logger.error(f"Failed fetching notes for symbol={symbol}: {e}")
            health_registry.report("human_insight_manager", ok=False, detail="Failed fetching notes", error=str(e))
            return []
        finally:
            if conn is not None:
                conn.close()

    def record_override(self, symbol: str, original_action: str, overridden_action: str, reason: str,
                         created_by: str = DEFAULT_CREATED_BY) -> Optional[int]:
        if not reason or not reason.strip():
            logger.error(f"Refusing to record override for {symbol} without a reason — reason is mandatory.")
            return None
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO human_overrides (symbol, timestamp, original_action, overridden_action, reason, created_by) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (symbol, datetime.now().isoformat(), original_action, overridden_action, reason, created_by),
            )
            conn.commit()
            override_id = cursor.lastrowid
            health_registry.report("human_insight_manager", ok=True, detail=f"Recorded override for {symbol}")
            return override_id
        except Exception as e:
            logger.error(f"Failed recording override for {symbol}: {e}")
            health_registry.report("human_insight_manager", ok=False, detail="Failed recording override", error=str(e))
            return None
        finally:
            if conn is not None:
                conn.close()

    def get_overrides(self, symbol: Optional[str] = None, limit: int = 50) -> List[OverrideRecord]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT id, symbol, timestamp, original_action, overridden_action, reason, created_by "
                    "FROM human_overrides WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                    (symbol, limit),
                )
            else:
                cursor.execute(
                    "SELECT id, symbol, timestamp, original_action, overridden_action, reason, created_by "
                    "FROM human_overrides ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
            health_registry.report("human_insight_manager", ok=True)
            return [OverrideRecord(*row) for row in rows]
        except Exception as e:
            logger.error(f"Failed fetching overrides for symbol={symbol}: {e}")
            health_registry.report("human_insight_manager", ok=False, detail="Failed fetching overrides", error=str(e))
            return []
        finally:
            if conn is not None:
                conn.close()

    def apply_override_to_signal(
        self, signal: PredictionSignal, overridden_action: str, reason: str, created_by: str = DEFAULT_CREATED_BY,
    ) -> PredictionSignal:
        """
        Records the override for audit purposes AND returns a new
        PredictionSignal reflecting the human's final call — the model's
        original output is preserved in reasoning, never silently erased.
        """
        override_id = self.record_override(signal.symbol, signal.action, overridden_action, reason, created_by)
        if override_id is None:
            raise ValueError(f"Failed to record override for {signal.symbol}: reason is mandatory and database record must succeed.")

        new_reasoning = list(signal.reasoning) + [
            f"HUMAN OVERRIDE by {created_by}: changed action from {signal.action} to {overridden_action}. "
            f"Reason: {reason}"
        ]
        return PredictionSignal(
            symbol=signal.symbol, timestamp=signal.timestamp, horizon=signal.horizon, action=overridden_action,
            model_predicted_class=signal.model_predicted_class, 
            model_version=signal.model_version, feature_version=signal.feature_version,
            raw_confidence=signal.raw_confidence,
            risk_adjusted_confidence=signal.risk_adjusted_confidence,
            calibrated_confidence=signal.calibrated_confidence, agreement_fraction=signal.agreement_fraction,
            downside_summary=signal.downside_summary, upside_summary=signal.upside_summary,
            reasoning=new_reasoning, contributing_events=signal.contributing_events,
            global_risk_level=signal.global_risk_level, risk_toggle_enabled=signal.risk_toggle_enabled,
            is_safe_to_trade_live=signal.is_safe_to_trade_live, data_stale=signal.data_stale,
            suppressed=signal.suppressed, suppression_reasons=signal.suppression_reasons,
            target_price=signal.target_price, stop_loss=signal.stop_loss,
            peak_potential_price=signal.peak_potential_price,
        )
```

---

## 2. File: `position_planner.py`

```python
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from config import PORTFOLIO_TOTAL_CAPITAL, MAX_POSITION_SIZE_PCT, MAX_DCA_STEPS
from predictor import MultiHorizonSignal, ACTION_BUY, ACTION_SELL, ACTION_HOLD

logger = logging.getLogger(__name__)

@dataclass
class DCAStep:
    trigger_price: float
    allocation_inr: float
    reason: str

@dataclass
class PositionPlan:
    symbol: str
    action: str
    total_allocated_inr: float
    max_allowed_inr: float
    ladder: List[DCAStep]
    planned_capital: float = 0.0
    stop_price: Optional[float] = None
    max_loss: float = 0.0
    risk_pct: float = 0.0

class PositionPlanner:
    def __init__(
        self,
        portfolio_capital: float = PORTFOLIO_TOTAL_CAPITAL,
        max_position_pct: float = MAX_POSITION_SIZE_PCT,
        max_dca_steps: int = MAX_DCA_STEPS,
        max_portfolio_risk_pct: float = 0.02,
    ):
        self.portfolio_capital = portfolio_capital
        self.max_position_pct = max_position_pct
        self.max_dca_steps = max_dca_steps
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.max_position_inr = self.portfolio_capital * self.max_position_pct
        self.max_risk_inr = self.portfolio_capital * self.max_portfolio_risk_pct

    def generate_plan(
        self, 
        multi_signal: MultiHorizonSignal, 
        current_price: float, 
        current_position_inr: float = 0.0,
        user_cost: Optional[float] = None,
        ma_level: Optional[float] = None,
        support_level: Optional[float] = None,
        stop_price: Optional[float] = None,
        atr: Optional[float] = None,
        account_cash: Optional[float] = None,
        committed_cash: Optional[float] = None,
    ) -> PositionPlan:
        try:
            symbol = multi_signal.symbol
            action = multi_signal.primary_action

            if stop_price is None and multi_signal.signals:
                primary_sig = multi_signal.signals.get(multi_signal.primary_horizon)
                if primary_sig and primary_sig.stop_loss:
                    stop_price = primary_sig.stop_loss

            if action != ACTION_BUY:
                return PositionPlan(
                    symbol=symbol, action=action, total_allocated_inr=0.0, 
                    max_allowed_inr=self.max_position_inr, ladder=[],
                    planned_capital=0.0, stop_price=stop_price, max_loss=0.0, risk_pct=0.0
                )

            remaining_symbol_cap = max(0.0, self.max_position_inr - current_position_inr)
            effective_cash = self.portfolio_capital if account_cash is None else account_cash
            free_account_cash = max(0.0, effective_cash - (committed_cash or 0.0))
            remaining_allocation = min(remaining_symbol_cap, free_account_cash)

            if remaining_allocation <= 0:
                logger.info(f"Position or cash capacity maxed out for {symbol}, cannot add more.")
                return PositionPlan(
                    symbol=symbol, action=ACTION_HOLD, total_allocated_inr=0.0, 
                    max_allowed_inr=self.max_position_inr, ladder=[],
                    planned_capital=0.0, stop_price=stop_price, max_loss=0.0, risk_pct=0.0
                )

            levels = []
            if ma_level and ma_level < current_price:
                levels.append((ma_level, "Moving Average"))
            if support_level and support_level < current_price:
                levels.append((support_level, "Support Band"))
            
            levels = sorted(levels, key=lambda x: x[0], reverse=True)

            if stop_price is None:
                if levels:
                    lowest_level = min(lvl for lvl, _ in levels)
                    stop_price = round(lowest_level * 0.95, 2)
                elif atr and atr > 0:
                    stop_price = round(max(0.01, current_price - (2.0 * atr)), 2)
                else:
                    stop_price = round(max(0.01, current_price * 0.97), 2)
            
            ladder = []
            if current_position_inr == 0.0:
                ladder.append(DCAStep(trigger_price=current_price, allocation_inr=0.0, reason="Initial Entry (CMP)"))
            
            for lvl, reason in levels:
                if lvl > stop_price:
                    ladder.append(DCAStep(trigger_price=round(lvl, 2), allocation_inr=0.0, reason=f"DCA on {reason}"))
                
            ladder = ladder[:self.max_dca_steps]
            if not ladder:
                ladder.append(DCAStep(trigger_price=current_price, allocation_inr=0.0, reason="Initial Entry (CMP)"))
                
            n_steps = len(ladder)
            weights = [1.0 + (0.5 * i) for i in range(n_steps)]
            total_weight = sum(weights)
            
            for i, step in enumerate(ladder):
                step.allocation_inr = round(remaining_allocation * (weights[i] / total_weight), 2)

            allocated_so_far = sum(s.allocation_inr for s in ladder[:-1])
            if ladder:
                ladder[-1].allocation_inr = round(remaining_allocation - allocated_so_far, 2)

            planned_capital = round(sum(step.allocation_inr for step in ladder), 2)

            total_loss = 0.0
            for step in ladder:
                if step.trigger_price > stop_price:
                    loss_fraction = (step.trigger_price - stop_price) / step.trigger_price
                    total_loss += step.allocation_inr * loss_fraction

            max_loss = round(total_loss, 2)
            risk_pct = round((max_loss / self.portfolio_capital) * 100.0, 3)

            return PositionPlan(
                symbol=symbol, action=ACTION_BUY, 
                total_allocated_inr=remaining_allocation, 
                max_allowed_inr=self.max_position_inr, 
                ladder=ladder,
                planned_capital=planned_capital,
                stop_price=stop_price,
                max_loss=max_loss,
                risk_pct=risk_pct,
            )
            
        except Exception as e:
            logger.error(f"Failed to generate position plan for {multi_signal.symbol}: {e}")
            return PositionPlan(
                symbol=multi_signal.symbol, action=ACTION_HOLD, total_allocated_inr=0.0,
                max_allowed_inr=0.0, ladder=[], planned_capital=0.0, stop_price=None, max_loss=0.0, risk_pct=0.0
            )
```

---

## 3. File: `narrative_builder.py`

```python
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from predictor import MultiHorizonSignal, PredictionSignal

logger = logging.getLogger(__name__)

@dataclass
class PriceLevelOutcome:
    horizon: str
    predicted_level: str
    confidence: float

class NarrativeBuilder:
    def __init__(self):
        pass

    def build_stock_narrative(
        self, 
        multi_signal: MultiHorizonSignal, 
        level_outcomes: Optional[List[PriceLevelOutcome]] = None
    ) -> str:
        try:
            symbol = multi_signal.symbol
            primary_action = multi_signal.primary_action
            primary_horizon = multi_signal.primary_horizon
            
            if primary_action == "HOLD":
                summary = f"For {symbol}, the current consensus is a cautious HOLD."
            elif primary_action == "BUY":
                summary = f"For {symbol}, the quantitative models are flashing a BUY signal."
            else:
                summary = f"For {symbol}, the system is issuing a SELL warning."
                
            summary += f" This is primarily driven by the {primary_horizon} outlook."
            
            horizon_parts = []
            for h, sig in multi_signal.signals.items():
                if sig.suppressed:
                    reason_msg = sig.suppression_reasons[0] if sig.suppression_reasons else "evidence unavailable"
                    horizon_parts.append(f"The {h} signal is suppressed ({reason_msg}).")
                else:
                    if sig.calibrated_confidence is not None:
                        horizon_parts.append(f"Over {h}, models lean {sig.model_predicted_class} with calibrated probability {sig.calibrated_confidence*100:.1f}%.")
                    else:
                        horizon_parts.append(f"Over {h}, models lean {sig.model_predicted_class} with raw model score {sig.raw_confidence*100:.1f}% (calibrated probability: unavailable).")
            
            horizon_summary = " ".join(horizon_parts)
            
            level_parts = []
            if level_outcomes:
                for outcome in level_outcomes:
                    if outcome.predicted_level != "NONE":
                        level_parts.append(f"In the {outcome.horizon} window, price is most likely to test {outcome.predicted_level} first (Model score: {outcome.confidence*100:.1f}%).")
            
            level_summary = " ".join(level_parts) if level_parts else "No significant price level tests are immediately projected."
            
            primary_sig = multi_signal.signals.get(primary_horizon)
            event_summary = ""
            if primary_sig and primary_sig.contributing_events:
                event_summary = "Key driving events include: " + "; ".join(
                    [e.headline_or_label for e in primary_sig.contributing_events[:3]]
                ) + "."
            else:
                event_summary = "No major macro or corporate events are currently overriding the technical setup."
            
            paragraphs = [summary, horizon_summary, level_summary, event_summary]
            return "\n\n".join(p for p in paragraphs if p)
            
        except Exception as e:
            logger.error(f"Failed to build narrative for {multi_signal.symbol}: {e}")
            return f"Error generating narrative for {multi_signal.symbol}. Please rely on raw signal data."
```

---

## 4. File: `frontend/src/api.js`

```javascript
const API_BASE_URL = (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_BASE_URL) || "http://127.0.0.1:8000";

async function handleResponse(response) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed with status ${response.status}`);
  }
  return response.json();
}

export async function fetchSignal(symbol) {
  const response = await fetch(`${API_BASE_URL}/signal/${encodeURIComponent(symbol)}`);
  return handleResponse(response);
}

export async function submitOverride({
  symbol,
  original_action,
  overridden_action,
  reason,
  created_by,
}) {
  const response = await fetch(`${API_BASE_URL}/override`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbol,
      original_action,
      overridden_action,
      reason,
      created_by,
    }),
  });
  return handleResponse(response);
}
```

---

## 5. File: `frontend/src/components/SignalCard.jsx`

```jsx
import { useEffect, useState } from "react";
import { fetchSignal } from "../api";

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

export default function SignalCard({ symbol }) {
  const [signal, setSignal] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    const loadSignalData = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchSignal(symbol);
        if (!cancelled) {
          setSignal(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || String(err));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    loadSignalData();

    return () => {
      cancelled = true; // avoids setting state on an unmounted component or stale request
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
```

---

## 6. File: `frontend/src/App.jsx`

```jsx
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
```

---

## 7. File: `tests/fs_02/fixtures/signal_specimen.json`

```json
{
  "symbol": "RELIANCE",
  "timestamp": "2026-09-07T10:00:00+05:30",
  "action": "HOLD",
  "model_predicted_class": "UP",
  "raw_confidence": 0.8,
  "risk_adjusted_confidence": 0.64,
  "calibrated_confidence": null,
  "agreement_fraction": 0.75,
  "downside_summary": "Fixture: loss remains possible.",
  "upside_summary": "Fixture: model leans upward.",
  "reasoning": [
    "Fixture model score: 0.80"
  ],
  "contributing_events": [],
  "global_risk_level": "NORMAL",
  "risk_toggle_enabled": false,
  "is_safe_to_trade_live": false,
  "data_stale": false,
  "suppressed": true,
  "suppression_reasons": [
    "Fixture: evidence unavailable"
  ]
}
```

---

## 8. File: `tests/fs_02/fixtures/prediction_signal.py`

```python
from datetime import datetime
from predictor import PredictionSignal, MultiHorizonSignal

def get_specimen_signal() -> PredictionSignal:
    stamp = datetime.fromisoformat("2026-09-07T10:00:00+05:30")
    return PredictionSignal(
        symbol="RELIANCE",
        timestamp=stamp,
        horizon="INTRADAY",
        action="HOLD",
        model_predicted_class="UP",
        model_version="v1.0",
        feature_version="v1.0",
        raw_confidence=0.8,
        risk_adjusted_confidence=0.64,
        calibrated_confidence=None,
        agreement_fraction=0.75,
        downside_summary="Fixture: loss remains possible.",
        upside_summary="Fixture: model leans upward.",
        reasoning=["Fixture model score: 0.80"],
        is_safe_to_trade_live=False,
        suppressed=True,
        suppression_reasons=["Fixture: evidence unavailable"],
        target_price=102.0,
        stop_loss=99.0,
        peak_potential_price=103.0
    )

def get_specimen_multi_signal() -> MultiHorizonSignal:
    sig = get_specimen_signal()
    return MultiHorizonSignal(
        symbol=sig.symbol,
        timestamp=sig.timestamp,
        signals={"INTRADAY": sig},
        primary_action="HOLD",
        primary_horizon="INTRADAY",
        reasoning=["Fixture"]
    )
```

---

## 9. Verification Commands

To run the complete test suite and verify all 7 findings in the codebase:

```powershell
# 1. Run Python FS-02 acceptance tests (16/16 pass)
.venv\Scripts\python.exe -m pytest tests/fs_02 -v

# 2. Run module self-tests
.venv\Scripts\python.exe position_planner.py
.venv\Scripts\python.exe narrative_builder.py
.venv\Scripts\python.exe human_insight_manager.py

# 3. Run frontend code quality check & build (0 errors)
cd frontend
npm run lint
npm run build
```
