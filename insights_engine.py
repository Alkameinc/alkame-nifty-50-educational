"""
insights_engine.py — Institutional Market Intelligence & Analytics Engine

STRICTLY EDUCATIONAL & INFORMATIONAL (SEBI COMPLIANT)
------------------------------------------------------
Under SEBI (Research Analysts) Regulations, 2014 and SEBI (Investment Advisers)
Regulations, 2013, this module computes and aggregates strictly factual market
data, descriptive mathematical technical indicators, corporate filings, macroeconomic
calendars, and public news sentiment.

This module explicitly DOES NOT:
- Provide BUY, SELL, or HOLD recommendations.
- Compute or publish price targets or stop-loss levels.
- Offer automated trade execution signals or scalping setups.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ALL_HORIZONS,
    BOLLINGER_PERIOD,
    BOLLINGER_STD_DEV,
    NIFTY50_SYMBOLS,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
    to_yfinance_ticker,
)
from corporate_events_fetcher import CorporateEventsFetcher
from data_fetcher import DataFetcher
from global_risk_monitor import GlobalRiskMonitor
from macro_calendar import MacroCalendar
from news_sentiment_fetcher import NewsSentimentFetcher
from reference_level_engine import ReferenceLevelEngine
from sector_provider import sector_map_provider

logger = logging.getLogger(__name__)

COMPLIANCE_DISCLAIMER = (
    "STATUTORY COMPLIANCE NOTICE: This market intelligence summary is compiled strictly "
    "for educational, academic, and informational analysis under SEBI guidelines. "
    "All figures represent historical calculations, public regulatory filings, and mathematical "
    "derivations. Alkame does not provide investment advice, trading recommendations, or "
    "price targets."
)


@dataclass
class TechnicalSnapshot:
    symbol: str
    cmp: float
    rsi_14: float
    rsi_status: str  # "OVERSOLD (<30)", "NEUTRAL (30-70)", "OVERBOUGHT (>70)"
    ema_20: float | None
    ema_50: float | None
    sma_200: float | None
    trend_posture: str  # "BULLISH_ALIGNMENT", "BEARISH_ALIGNMENT", "NEUTRAL_MIXED"
    bollinger_upper: float | None
    bollinger_middle: float | None
    bollinger_lower: float | None
    bollinger_pct_b: float | None
    bollinger_bandwidth: float | None
    atr_14: float | None
    atr_pct: float | None
    volume_latest: int
    volume_20d_sma: float | None
    volume_ratio: float | None
    volume_surge: bool
    vwap: float | None
    vwap_relation: str  # "PREMIUM" | "DISCOUNT" | "AT_PAR"
    pivots_camarilla: dict[str, float] = field(default_factory=dict)
    pivots_standard: dict[str, float] = field(default_factory=dict)
    prior_day_high: float | None = None
    prior_day_low: float | None = None
    prior_day_close: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    pct_from_52w_high: float | None = None
    pct_from_52w_low: float | None = None


@dataclass
class MarketInsightsResult:
    symbol: str
    timestamp: str
    cmp: float
    day_change_pct: float
    technical: TechnicalSnapshot
    valuation: dict[str, Any]
    corporate_events: list[dict[str, Any]]
    macro_context: dict[str, Any]
    news_sentiment: dict[str, Any]
    disclaimer: str = COMPLIANCE_DISCLAIMER


class MarketInsightsEngine:
    """
    Central aggregation engine that unifies multi-source market intelligence
    into a comprehensive, compliant analytical payload for institutional research.
    """

    def __init__(
        self,
        data_fetcher: DataFetcher | None = None,
        corporate_events_fetcher: CorporateEventsFetcher | None = None,
        macro_calendar: MacroCalendar | None = None,
        news_sentiment_fetcher: NewsSentimentFetcher | None = None,
        global_risk_monitor: GlobalRiskMonitor | None = None,
        reference_level_engine: ReferenceLevelEngine | None = None,
    ):
        self.data_fetcher = data_fetcher or DataFetcher()
        self.corporate_events_fetcher = corporate_events_fetcher or CorporateEventsFetcher()
        self.macro_calendar = macro_calendar or MacroCalendar()
        self.news_sentiment_fetcher = news_sentiment_fetcher or NewsSentimentFetcher()
        self.global_risk_monitor = global_risk_monitor or GlobalRiskMonitor(data_fetcher=self.data_fetcher)
        self.reference_level_engine = reference_level_engine or ReferenceLevelEngine()

    def compute_technical_snapshot(self, symbol: str, stock_df: pd.DataFrame) -> TechnicalSnapshot:
        """
        Computes factual technical indicators from historical OHLCV data with no lookahead bias.
        """
        if stock_df is None or stock_df.empty or len(stock_df) < 20:
            cmp = float(stock_df["Close"].iloc[-1]) if stock_df is not None and not stock_df.empty else 0.0
            return TechnicalSnapshot(
                symbol=symbol,
                cmp=cmp,
                rsi_14=50.0,
                rsi_status="NEUTRAL (30-70)",
                ema_20=None,
                ema_50=None,
                sma_200=None,
                trend_posture="INSUFFICIENT_DATA",
                bollinger_upper=None,
                bollinger_middle=None,
                bollinger_lower=None,
                bollinger_pct_b=None,
                bollinger_bandwidth=None,
                atr_14=None,
                atr_pct=None,
                volume_latest=0,
                volume_20d_sma=None,
                volume_ratio=None,
                volume_surge=False,
                vwap=None,
                vwap_relation="AT_PAR",
            )

        closes = stock_df["Close"]
        highs = stock_df["High"]
        lows = stock_df["Low"]
        volumes = stock_df["Volume"]
        cmp = float(closes.iloc[-1])

        # 1. RSI-14
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean()
        avg_loss = loss.rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))
        rsi_val = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0

        if rsi_val >= RSI_OVERBOUGHT:
            rsi_status = f"OVERBOUGHT (>={RSI_OVERBOUGHT})"
        elif rsi_val <= RSI_OVERSOLD:
            rsi_status = f"OVERSOLD (<={RSI_OVERSOLD})"
        else:
            rsi_status = "NEUTRAL (30-70)"

        # 2. Moving Averages
        ema_20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1]) if len(closes) >= 20 else None
        ema_50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1]) if len(closes) >= 50 else None
        sma_200 = float(closes.rolling(window=200).mean().iloc[-1]) if len(closes) >= 200 else None

        trend_posture = "NEUTRAL_MIXED"
        if ema_20 and ema_50:
            if cmp > ema_20 > ema_50:
                trend_posture = "BULLISH_ALIGNMENT"
            elif cmp < ema_20 < ema_50:
                trend_posture = "BEARISH_ALIGNMENT"

        # 3. Bollinger Bands (20, 2)
        bb_mid = float(closes.rolling(window=BOLLINGER_PERIOD).mean().iloc[-1]) if len(closes) >= BOLLINGER_PERIOD else None
        bb_std = float(closes.rolling(window=BOLLINGER_PERIOD).std().iloc[-1]) if len(closes) >= BOLLINGER_PERIOD else None
        if bb_mid is not None and bb_std is not None:
            bb_upper = bb_mid + (BOLLINGER_STD_DEV * bb_std)
            bb_lower = bb_mid - (BOLLINGER_STD_DEV * bb_std)
            bb_pct_b = (cmp - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            bb_bandwidth = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0.0
        else:
            bb_upper, bb_lower, bb_pct_b, bb_bandwidth = None, None, None, None

        # 4. ATR (14)
        tr1 = highs - lows
        tr2 = (highs - closes.shift()).abs()
        tr3 = (lows - closes.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = float(tr.rolling(window=14).mean().iloc[-1]) if len(tr) >= 14 else None
        atr_pct = (atr_14 / cmp * 100.0) if (atr_14 and cmp > 0) else None

        # 5. Volume Analysis
        vol_latest = int(volumes.iloc[-1])
        vol_20d_sma = float(volumes.rolling(window=20).mean().iloc[-1]) if len(volumes) >= 20 else None
        if vol_20d_sma and vol_20d_sma > 0:
            vol_ratio = vol_latest / vol_20d_sma
            vol_surge = vol_ratio >= 1.5
        else:
            vol_ratio = 1.0
            vol_surge = False

        # 6. VWAP (Intraday or rolling standard)
        typical_price = (highs + lows + closes) / 3.0
        vwap_num = (typical_price * volumes).cumsum()
        vwap_den = volumes.cumsum()
        vwap = float((vwap_num / vwap_den.replace(0, np.nan)).iloc[-1]) if not vwap_den.empty else None

        if vwap:
            if cmp > vwap * 1.002:
                vwap_relation = "PREMIUM (Above VWAP)"
            elif cmp < vwap * 0.998:
                vwap_relation = "DISCOUNT (Below VWAP)"
            else:
                vwap_relation = "AT_PAR"
        else:
            vwap_relation = "UNAVAILABLE"

        # 7. Reference Levels & Pivots (Camarilla & Standard)
        prev_h = float(highs.iloc[-2]) if len(highs) >= 2 else cmp
        prev_l = float(lows.iloc[-2]) if len(lows) >= 2 else cmp
        prev_c = float(closes.iloc[-2]) if len(closes) >= 2 else cmp
        diff = prev_h - prev_l

        camarilla = {
            "H4": round(prev_c + diff * 1.1 / 2.0, 2),
            "H3": round(prev_c + diff * 1.1 / 4.0, 2),
            "L3": round(prev_c - diff * 1.1 / 4.0, 2),
            "L4": round(prev_c - diff * 1.1 / 2.0, 2),
        }

        std_pivot = (prev_h + prev_l + prev_c) / 3.0
        standard_pivots = {
            "P": round(std_pivot, 2),
            "R1": round(2 * std_pivot - prev_l, 2),
            "S1": round(2 * std_pivot - prev_h, 2),
            "R2": round(std_pivot + (prev_h - prev_l), 2),
            "S2": round(std_pivot - (prev_h - prev_l), 2),
        }

        # 8. 52-Week Range
        high_52w = float(highs.max())
        low_52w = float(lows.min())
        pct_from_high = ((cmp - high_52w) / high_52w) * 100.0 if high_52w > 0 else 0.0
        pct_from_low = ((cmp - low_52w) / low_52w) * 100.0 if low_52w > 0 else 0.0

        return TechnicalSnapshot(
            symbol=symbol,
            cmp=round(cmp, 2),
            rsi_14=round(rsi_val, 1),
            rsi_status=rsi_status,
            ema_20=round(ema_20, 2) if ema_20 else None,
            ema_50=round(ema_50, 2) if ema_50 else None,
            sma_200=round(sma_200, 2) if sma_200 else None,
            trend_posture=trend_posture,
            bollinger_upper=round(bb_upper, 2) if bb_upper else None,
            bollinger_middle=round(bb_mid, 2) if bb_mid else None,
            bollinger_lower=round(bb_lower, 2) if bb_lower else None,
            bollinger_pct_b=round(bb_pct_b, 2) if bb_pct_b is not None else None,
            bollinger_bandwidth=round(bb_bandwidth, 4) if bb_bandwidth is not None else None,
            atr_14=round(atr_14, 2) if atr_14 else None,
            atr_pct=round(atr_pct, 2) if atr_pct else None,
            volume_latest=vol_latest,
            volume_20d_sma=round(vol_20d_sma, 0) if vol_20d_sma else None,
            volume_ratio=round(vol_ratio, 2) if vol_ratio else None,
            volume_surge=vol_surge,
            vwap=round(vwap, 2) if vwap else None,
            vwap_relation=vwap_relation,
            pivots_camarilla=camarilla,
            pivots_standard=standard_pivots,
            prior_day_high=round(prev_h, 2),
            prior_day_low=round(prev_l, 2),
            prior_day_close=round(prev_c, 2),
            fifty_two_week_high=round(high_52w, 2),
            fifty_two_week_low=round(low_52w, 2),
            pct_from_52w_high=round(pct_from_high, 1),
            pct_from_52w_low=round(pct_from_low, 1),
        )

    def get_stock_insights(self, symbol: str) -> MarketInsightsResult:
        """
        Gathers complete educational and informational intelligence for a given NIFTY 50 symbol.
        """
        yf_ticker = to_yfinance_ticker(symbol)

        # 1. Market Data
        raw_stock = self.data_fetcher.fetch_ohlcv(yf_ticker, allow_stale=True)
        stock_df = getattr(raw_stock, "df", raw_stock)
        if stock_df is None or stock_df.empty:
            cmp = 0.0
            day_chg = 0.0
        else:
            cmp = float(stock_df["Close"].iloc[-1])
            prev_close = float(stock_df["Close"].iloc[-2]) if len(stock_df) >= 2 else cmp
            day_chg = ((cmp - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0

        # 2. Technical Snapshot
        technical = self.compute_technical_snapshot(symbol, stock_df)

        # 3. Fundamental Valuation
        fundamentals = self.data_fetcher.fetch_stock_fundamentals(yf_ticker)
        if not fundamentals.get("sector") or fundamentals.get("sector") == "N/A":
            sector_info = sector_map_provider.get_symbol_sector(symbol)
            fundamentals["sector"] = sector_info.get("sector", "N/A")
            fundamentals["industry"] = sector_info.get("industry", "N/A")

        # 4. Corporate Events & Filings (from NSE)
        try:
            corp_events_raw = self.corporate_events_fetcher.fetch_all_for_symbol(symbol)
            corporate_events = [
                {
                    "category": e.category,
                    "date": e.date.isoformat() if hasattr(e.date, "isoformat") else str(e.date),
                    "headline": e.headline,
                    "details": e.details,
                }
                for e in corp_events_raw[:10]
            ]
        except Exception as e:
            logger.warning(f"Could not fetch corporate events for {symbol}: {e}")
            corporate_events = []

        # 5. Macro Risk Context
        try:
            risk_reading = self.global_risk_monitor.compute_composite_risk()
            all_macro = self.macro_calendar.get_all_events()
            macro_list = []
            if all_macro:
                for item in all_macro:
                    if hasattr(item, "date"):
                        macro_list.append({
                            "date": str(item.date),
                            "event_type": str(item.event_type),
                            "label": str(item.label),
                        })
                    elif isinstance(item, dict):
                        macro_list.append({
                            "date": str(item.get("date", "")),
                            "event_type": str(item.get("event_type", "")),
                            "label": str(item.get("label", "")),
                        })
            macro_context = {
                "composite_risk_level": risk_reading.risk_level,
                "dominant_driver": risk_reading.dominant_driver,
                "banner_message": risk_reading.banner_message,
                "upcoming_macro_events": macro_list[:5],
            }
        except Exception as e:
            logger.warning(f"Could not compute macro context: {e}")
            macro_context = {"composite_risk_level": "NORMAL", "dominant_driver": None, "upcoming_macro_events": []}

        # 6. News Sentiment Analysis (Factual Polarity)
        try:
            news_items = self.news_sentiment_fetcher.get_news_for_symbol(symbol)
            if news_items:
                scores = [item["sentiment_score"] for item in news_items if item.get("sentiment_score") is not None]
                avg_score = float(np.mean(scores)) if scores else 0.0
                news_sentiment = {
                    "sentiment_score": round(avg_score, 2),
                    "sentiment_label": "POSITIVE" if avg_score > 0.15 else ("CAUTIOUS" if avg_score < -0.15 else "NEUTRAL"),
                    "articles_analyzed": len(news_items),
                    "recent_headlines": [
                        {
                            "title": item.get("title", ""),
                            "source": item.get("source", ""),
                            "published_at": str(item.get("published_at", "")),
                            "sentiment": item.get("sentiment_score", 0.0),
                        }
                        for item in news_items[:5]
                    ],
                }
            else:
                news_sentiment = {"sentiment_score": 0.0, "sentiment_label": "NEUTRAL", "articles_analyzed": 0, "recent_headlines": []}
        except Exception as e:
            logger.warning(f"Could not fetch news sentiment for {symbol}: {e}")
            news_sentiment = {"sentiment_score": 0.0, "sentiment_label": "NEUTRAL", "articles_analyzed": 0, "recent_headlines": []}

        return MarketInsightsResult(
            symbol=symbol,
            timestamp=datetime.now(UTC).isoformat(),
            cmp=round(cmp, 2),
            day_change_pct=round(day_chg, 2),
            technical=technical,
            valuation=fundamentals,
            corporate_events=corporate_events,
            macro_context=macro_context,
            news_sentiment=news_sentiment,
        )

    def get_market_breadth(self) -> dict[str, Any]:
        """
        Scans NIFTY 50 constituents to produce a factual market breadth and volume activity radar.
        Identifies stocks with volume surges and proximity to 52-week extremes without recommendations.
        """
        advances = 0
        declines = 0
        unchanged = 0
        constituent_summaries = []

        for symbol in NIFTY50_SYMBOLS:
            try:
                yf_ticker = to_yfinance_ticker(symbol)
                raw_stock = self.data_fetcher.fetch_ohlcv(yf_ticker, allow_stale=True)
                stock_df = getattr(raw_stock, "df", raw_stock)
                if stock_df is None or stock_df.empty or len(stock_df) < 5:
                    continue

                cmp = float(stock_df["Close"].iloc[-1])
                prev = float(stock_df["Close"].iloc[-2]) if len(stock_df) >= 2 else cmp
                chg_pct = ((cmp - prev) / prev * 100.0) if prev > 0 else 0.0

                if chg_pct > 0.05:
                    advances += 1
                elif chg_pct < -0.05:
                    declines += 1
                else:
                    unchanged += 1

                # Volume Surge Check
                volumes = stock_df["Volume"]
                vol_latest = int(volumes.iloc[-1])
                vol_sma = float(volumes.rolling(20).mean().iloc[-1]) if len(volumes) >= 20 else vol_latest
                vol_ratio = vol_latest / vol_sma if vol_sma > 0 else 1.0

                # 52-week High/Low
                h52 = float(stock_df["High"].max())
                l52 = float(stock_df["Low"].min())
                pct_from_high = ((cmp - h52) / h52) * 100.0 if h52 > 0 else 0.0

                # Quick RSI
                delta = stock_df["Close"].diff()
                g = delta.clip(lower=0).rolling(14).mean()
                l = (-delta.clip(upper=0)).rolling(14).mean()
                rs = g / l.replace(0, np.nan)
                rsi = float((100 - (100 / (1 + rs))).iloc[-1]) if not pd.isna(rs.iloc[-1]) else 50.0

                sector_info = sector_map_provider.get_symbol_sector(symbol)

                constituent_summaries.append({
                    "symbol": symbol,
                    "cmp": round(cmp, 2),
                    "day_change_pct": round(chg_pct, 2),
                    "volume": vol_latest,
                    "volume_ratio": round(vol_ratio, 2),
                    "volume_surge": vol_ratio >= 1.5,
                    "rsi_14": round(rsi, 1),
                    "pct_from_52w_high": round(pct_from_high, 1),
                    "sector": sector_info.get("sector", "Other"),
                })
            except Exception as e:
                logger.debug(f"Error scanning breadth for {symbol}: {e}")

        # Sort volume surges
        volume_surges = sorted(
            [c for c in constituent_summaries if c["volume_surge"]],
            key=lambda x: x["volume_ratio"],
            reverse=True,
        )

        top_gainers = sorted(constituent_summaries, key=lambda x: x["day_change_pct"], reverse=True)[:5]
        top_losers = sorted(constituent_summaries, key=lambda x: x["day_change_pct"])[:5]

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_constituents": len(constituent_summaries),
            "advances": advances,
            "declines": declines,
            "unchanged": unchanged,
            "advance_decline_ratio": round(advances / declines, 2) if declines > 0 else advances,
            "volume_surges": volume_surges,
            "top_gainers": top_gainers,
            "top_losers": top_losers,
            "all_constituents": constituent_summaries,
            "disclaimer": COMPLIANCE_DISCLAIMER,
        }


# Singleton engine instance
insights_engine = MarketInsightsEngine()
