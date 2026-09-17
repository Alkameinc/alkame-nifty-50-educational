# 1. Standard library imports
from datetime import date, datetime
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Third-party imports
import numpy as np
import pandas as pd
import pytest

# 3. Local imports
from backtester import BacktestResult, Backtester
from config import HORIZON_INTRADAY
from universe_provider import UniverseProvider, get_nifty50_constituents, universe_provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_synthetic_intraday_ohlcv(
    start_date: str = "2020-01-06",
    n_days: int = 25,
    bars_per_day: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic intraday bars (1000 bars total) with sufficient history for training."""
    rng = np.random.default_rng(seed)
    rows, timestamps = [], []
    price = 1000.0
    base = pd.Timestamp(start_date)
    recent_closes: list[float] = []
    for day in range(n_days):
        day_start = base + pd.Timedelta(days=day)
        for bar in range(bars_per_day):
            ts = day_start + pd.Timedelta(minutes=5 * bar)
            if len(recent_closes) >= 10:
                trend = recent_closes[-1] - recent_closes[-10]
                bias = 1.5 if trend < -6 else (-1.5 if trend > 6 else 0.0)
            else:
                bias = 0.0
            drift = rng.normal(bias, 1.5)
            price = max(10.0, price + drift)
            open_p = price
            close_p = max(10.0, price + rng.normal(bias * 0.5, 1.0))
            high_p = max(open_p, close_p) + abs(rng.normal(0, 0.5))
            low_p = min(open_p, close_p) - abs(rng.normal(0, 0.5))
            vol = int(abs(rng.normal(50000, 15000)))
            rows.append([open_p, high_p, low_p, close_p, vol])
            timestamps.append(ts)
            recent_closes.append(close_p)
    return pd.DataFrame(
        rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(timestamps)
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
class TestPhase10SurvivorshipBias:
    """
    Phase 10: Survivorship Bias Remediation (DATA-006 & BACK-004)
    """

    def test_dataset_integrity_and_continuity(self):
        """DATA-006: Verify historical membership intervals maintain exactly 50 unique constituents."""
        provider = UniverseProvider()
        history = provider.get_membership_history()
        assert len(history) >= 50, "Expected at least 50 historical membership records"

        # Check milestones across 2020-2025
        check_dates = [
            "2020-03-15",
            "2020-09-24",
            "2020-09-25",
            "2021-03-30",
            "2021-03-31",
            "2022-03-30",
            "2022-03-31",
            "2022-09-29",
            "2022-09-30",
            "2023-07-12",
            "2023-07-13",
            "2024-03-27",
            "2024-03-28",
            "2024-09-29",
            "2024-09-30",
            "2025-01-15",
        ]
        for d in check_dates:
            univ = provider.eligible_universe(d)
            assert len(univ) == 50, f"Date {d} expected 50 constituents, got {len(univ)}"
            assert len(set(univ)) == 50, f"Date {d} contains duplicate constituents"
            assert provider.validate_constituents(50, as_of_date=d) is True

    def test_eligible_universe_point_in_time_milestones(self):
        """BACK-004: Point-in-time constituent resolution across all historical reconstitution events."""
        provider = UniverseProvider()

        # 1. September 2020: ZEEL & INFRATEL out, DIVISLAB & SBILIFE in
        pre_sep_2020 = provider.eligible_universe("2020-06-01")
        post_sep_2020 = provider.eligible_universe("2020-10-01")
        assert "ZEEL" in pre_sep_2020 and "INFRATEL" in pre_sep_2020
        assert "DIVISLAB" not in pre_sep_2020 and "SBILIFE" not in pre_sep_2020
        assert "ZEEL" not in post_sep_2020 and "INFRATEL" not in post_sep_2020
        assert "DIVISLAB" in post_sep_2020 and "SBILIFE" in post_sep_2020

        # 2. March 2021: GAIL out, TATACONSUM in
        pre_mar_2021 = provider.eligible_universe("2021-02-15")
        post_mar_2021 = provider.eligible_universe("2021-04-15")
        assert "GAIL" in pre_mar_2021 and "TATACONSUM" not in pre_mar_2021
        assert "GAIL" not in post_mar_2021 and "TATACONSUM" in post_mar_2021

        # 3. March 2022: IOC out, APOLLOHOSP in
        pre_mar_2022 = provider.eligible_universe("2022-02-15")
        post_mar_2022 = provider.eligible_universe("2022-04-15")
        assert "IOC" in pre_mar_2022 and "APOLLOHOSP" not in pre_mar_2022
        assert "IOC" not in post_mar_2022 and "APOLLOHOSP" in post_mar_2022

        # 4. September 2022: SHREECEM out, ADANIENT in
        pre_sep_2022 = provider.eligible_universe("2022-08-15")
        post_sep_2022 = provider.eligible_universe("2022-10-15")
        assert "SHREECEM" in pre_sep_2022 and "ADANIENT" not in pre_sep_2022
        assert "SHREECEM" not in post_sep_2022 and "ADANIENT" in post_sep_2022

        # 5. July 2023: HDFC out (merger), LTIM in
        pre_jul_2023 = provider.eligible_universe("2023-06-15")
        post_jul_2023 = provider.eligible_universe("2023-08-15")
        assert "HDFC" in pre_jul_2023 and "LTIM" not in pre_jul_2023
        assert "HDFC" not in post_jul_2023 and "LTIM" in post_jul_2023

        # 6. March 2024: UPL out, SHRIRAMFIN in
        pre_mar_2024 = provider.eligible_universe("2024-02-15")
        post_mar_2024 = provider.eligible_universe("2024-04-15")
        assert "UPL" in pre_mar_2024 and "SHRIRAMFIN" not in pre_mar_2024
        assert "UPL" not in post_mar_2024 and "SHRIRAMFIN" in post_mar_2024

        # 7. September 2024: DIVISLAB & LTIM out, BEL & TRENT in
        pre_sep_2024 = provider.eligible_universe("2024-08-15")
        post_sep_2024 = provider.eligible_universe("2024-10-15")
        assert "DIVISLAB" in pre_sep_2024 and "LTIM" in pre_sep_2024
        assert "BEL" not in pre_sep_2024 and "TRENT" not in pre_sep_2024
        assert "DIVISLAB" not in post_sep_2024 and "LTIM" not in post_sep_2024
        assert "BEL" in post_sep_2024 and "TRENT" in post_sep_2024

    def test_is_member_and_date_type_flexibility(self):
        """BACK-004: Verify is_member supports strings, dates, datetimes, and Timestamps with case insensitivity."""
        provider = UniverseProvider()

        # String
        assert provider.is_member("gail", "2020-05-01") is True
        assert provider.is_member("GAIL", "2022-05-01") is False

        # datetime.date
        assert provider.is_member("TATACONSUM", date(2020, 5, 1)) is False
        assert provider.is_member("tataconsum", date(2022, 5, 1)) is True

        # datetime.datetime
        assert provider.is_member("UPL", datetime(2023, 1, 1, 9, 15)) is True
        assert provider.is_member("UPL", datetime(2024, 6, 1, 9, 15)) is False

        # pd.Timestamp
        assert provider.is_member("SHRIRAMFIN", pd.Timestamp("2023-01-01 09:15:00")) is False
        assert provider.is_member("SHRIRAMFIN", pd.Timestamp("2024-06-01 09:15:00")) is True

        # Nonexistent ticker
        assert provider.is_member("NONEXISTENT_TICKER", "2024-01-01") is False

    def test_membership_changes_and_history_queries(self):
        """DATA-006: Verify get_membership_changes and get_membership_history."""
        provider = UniverseProvider()

        # Filter history by symbol
        upl_history = provider.get_membership_history("UPL")
        assert len(upl_history) == 1
        assert upl_history[0].symbol == "UPL"
        assert upl_history[0].effective_to == "2024-03-27"

        # Changes between 2024-01-01 and 2024-12-31
        changes_2024 = provider.get_membership_changes(start_date="2024-01-01", end_date="2024-12-31")
        change_syms = {c["symbol"] for c in changes_2024}
        assert "UPL" in change_syms  # Excluded
        assert "SHRIRAMFIN" in change_syms  # Included
        assert "BEL" in change_syms  # Included
        assert "TRENT" in change_syms  # Included
        assert "DIVISLAB" in change_syms  # Excluded
        assert "LTIM" in change_syms  # Excluded

    def test_backtester_enforces_universe_during_simulation(self):
        """BACK-004: Verify Backtester suppresses trades when enforce_universe=True for non-member dates."""
        backtester = Backtester()

        # Build 1000 intraday bars for 2020 (when TATACONSUM was NOT in NIFTY 50)
        stock_df = _build_synthetic_intraday_ohlcv(start_date="2020-01-06", n_days=25, bars_per_day=40, seed=10)
        index_df = _build_synthetic_intraday_ohlcv(start_date="2020-01-06", n_days=25, bars_per_day=40, seed=20)
        index_df.index = stock_df.index

        # Run with enforce_universe=False (survivorship-biased)
        res_biased = backtester.run_backtest_for_symbol(
            symbol="TATACONSUM",
            stock_df=stock_df,
            index_df=index_df,
            horizon=HORIZON_INTRADAY,
            enforce_universe=False,
        )
        assert res_biased.success is True, f"Biased run failed: {res_biased.error}"
        assert res_biased.enforce_universe is False
        assert res_biased.n_universe_excluded_bars == 0
        assert res_biased.n_trades_taken > 0, "Biased backtest took 0 trades"

        # Run with enforce_universe=True (survivorship-bias-free)
        res_bias_free = backtester.run_backtest_for_symbol(
            symbol="TATACONSUM",
            stock_df=stock_df,
            index_df=index_df,
            horizon=HORIZON_INTRADAY,
            enforce_universe=True,
        )
        assert res_bias_free.success is True, f"Bias-free run failed: {res_bias_free.error}"
        assert res_bias_free.enforce_universe is True
        assert res_bias_free.n_universe_excluded_bars > 0
        assert res_bias_free.n_trades_taken == 0, (
            "Expected 0 trades taken because TATACONSUM was not in NIFTY 50 during 2020"
        )
        assert res_bias_free.strategy_cumulative_return_pct == 0.0

    def test_replay_point_in_time_signal_universe_gate(self):
        """BACK-004: Point-in-time replay suppresses signals when symbol was not in NIFTY 50 at T."""
        backtester = Backtester()
        stock_df = _build_synthetic_intraday_ohlcv(start_date="2020-01-06", n_days=25, bars_per_day=40, seed=15)
        index_df = _build_synthetic_intraday_ohlcv(start_date="2020-01-06", n_days=25, bars_per_day=40, seed=25)
        index_df.index = stock_df.index

        # Train model so predictor can generate signal
        backtester.ensemble_manager.train_ensemble_for_symbol("TATACONSUM", stock_df, index_df, horizon=HORIZON_INTRADAY)

        ts = stock_df.index[-1]

        # 1. Without enforce_universe
        sig_unconstrained = backtester.replay_point_in_time_signal(
            symbol="TATACONSUM",
            timestamp=ts,
            stock_df=stock_df,
            index_df=index_df,
            enforce_universe=False,
        )
        # 2. With enforce_universe=True at 2020 timestamp
        sig_constrained = backtester.replay_point_in_time_signal(
            symbol="TATACONSUM",
            timestamp=ts,
            stock_df=stock_df,
            index_df=index_df,
            enforce_universe=True,
        )

        assert sig_constrained.action == "HOLD"
        assert sig_constrained.suppressed is True
        all_reasons = (getattr(sig_constrained, "suppression_reasons", []) or []) + (getattr(sig_constrained, "reasoning", []) or [])
        assert any("SURVIVORSHIP_BIAS_GATE" in reason for reason in all_reasons)

    def test_multi_asset_universe_backtest_quantifies_survivorship_bias(self):
        """BACK-004: Quantifies portfolio-level distortion caused by survivorship bias."""
        backtester = Backtester()

        # Create 2 symbols:
        # RELIANCE: Core member throughout 2020
        # TATACONSUM: NOT a member in 2020
        rel_df = _build_synthetic_intraday_ohlcv(start_date="2020-01-06", n_days=25, bars_per_day=40, seed=30)
        tataconsum_df = _build_synthetic_intraday_ohlcv(start_date="2020-01-06", n_days=25, bars_per_day=40, seed=40)
        idx_df = _build_synthetic_intraday_ohlcv(start_date="2020-01-06", n_days=25, bars_per_day=40, seed=50)
        tataconsum_df.index = rel_df.index
        idx_df.index = rel_df.index

        symbol_data = {
            "RELIANCE": (rel_df, idx_df),
            "TATACONSUM": (tataconsum_df, idx_df),
        }

        quant_report = backtester.quantify_survivorship_bias(symbol_data, horizon=HORIZON_INTRADAY)

        assert "biased_run" in quant_report
        assert "bias_free_run" in quant_report
        assert "survivorship_bias_delta" in quant_report

        delta = quant_report["survivorship_bias_delta"]
        # TATACONSUM bars were excluded in bias-free run
        assert delta["universe_excluded_bars"] > 0
        assert delta["phantom_trades_count"] > 0
        assert "summary" in quant_report

    def test_backwards_compatibility_and_config_sync(self):
        """DATA-006: Verify get_constituents() with no args still returns active 50 constituents."""
        provider = UniverseProvider()
        active = provider.get_constituents()
        assert len(active) == 50
        assert len(set(active)) == 50

        # Global function
        global_active = get_nifty50_constituents()
        assert len(global_active) == 50
        assert global_active == active

        # Active snapshot matches
        snap = provider.get_snapshot()
        assert snap is not None
        assert snap.total_constituents == 50
        assert set(snap.constituents) == set(active)
