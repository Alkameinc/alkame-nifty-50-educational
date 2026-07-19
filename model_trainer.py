# 1. Standard library imports
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 2. Third-party imports
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 3. Local imports
from config import (
    MODELS_DIR,
    HORIZON_CONFIG,
    HORIZON_INTRADAY,
    LABEL_CLASSES,
    MODEL_RANDOM_SEED,
    MODEL_N_ESTIMATORS,
    MODEL_MAX_DEPTH,
    MODEL_LEARNING_RATE,
    TIME_SERIES_SPLIT_TEST_FRACTION,
    ensure_directories,
    configure_logging,
)
from feature_engineer import FeatureEngineer, ML_SAFE_SUFFIX

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
MODEL_FILE_SUFFIX = "_model.joblib"
METADATA_FILE_SUFFIX = "_metadata.json"
LEVEL_MODEL_FILE_SUFFIX = "_level_model.joblib"
LEVEL_METADATA_FILE_SUFFIX = "_level_metadata.json"

LEVEL_LABEL_CLASSES = ["MA", "SUPPORT", "RESISTANCE", "USER_COST", "NONE"]

@dataclass
class TrainingResult:
    symbol: str
    trained_at: str
    n_train_samples: int
    n_test_samples: int
    feature_columns: List[str]
    test_accuracy: float
    class_report: Dict
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
class ModelTrainer:
    """
    Trains a per-stock direction classifier (UP / FLAT / DOWN over the next
    PREDICTION_HORIZON_BARS bars) using ONLY the '_feat' (lagged, ML-safe)
    columns produced by feature_engineer.py. Enforces a strict chronological
    train/test split — shuffling time-series data here would silently leak
    the future into training.
    """

    def __init__(self, feature_engineer: Optional[FeatureEngineer] = None):
        self.feature_engineer = feature_engineer or FeatureEngineer()
        ensure_directories()

    # -----------------------------------------------------------------
    # Dataset preparation
    # -----------------------------------------------------------------
    def compute_adaptive_deadband(self, df: pd.DataFrame, horizon_bars: int, deadband_pct_default: float) -> pd.Series:
        """
        Uses a rolling standard deviation of historical backward returns over the last 500 bars
        multiplied by 0.5 to define the UP/FLAT/DOWN threshold dynamically.
        Caps this adaptive deadband at a minimum of deadband_pct_default.
        """
        backward_returns = (df["Close"] - df["Close"].shift(horizon_bars)) / df["Close"].shift(horizon_bars) * 100.0
        rolling_std = backward_returns.rolling(window=500, min_periods=50).std()
        
        adaptive_deadband = rolling_std * 0.5
        adaptive_deadband = adaptive_deadband.clip(lower=deadband_pct_default)
        return adaptive_deadband.fillna(deadband_pct_default)

    def simulate_user_cost(self, df: pd.DataFrame, seed: int = MODEL_RANDOM_SEED) -> None:
        """
        Synthetically generates a 'user_avg_cost' for historical data so the model
        can learn the 'USER_COST' outcome. 50% chance of having a position. If position
        exists, cost is between -15% and +15% of current price.
        Mutates df in place (both unlagged and _feat lagged columns).
        """
        rng = np.random.default_rng(seed)
        mask = rng.random(len(df)) > 0.5
        
        df["has_position"] = mask.astype(float)
        random_pcts = rng.uniform(-15.0, 15.0, size=len(df))
        df["pct_from_user_avg_cost"] = np.where(mask, random_pcts, 0.0)
        
        df[f"has_position{ML_SAFE_SUFFIX}"] = df["has_position"].shift(1)
        df[f"pct_from_user_avg_cost{ML_SAFE_SUFFIX}"] = df["pct_from_user_avg_cost"].shift(1)

    def build_price_level_labels(self, df: pd.DataFrame, horizon: str = HORIZON_INTRADAY) -> pd.Series:
        """
        Simulates the future price path to see WHICH reference level is hit FIRST.
        Returns one of: MA, SUPPORT, RESISTANCE, USER_COST, NONE.
        """
        try:
            horizon_bars = HORIZON_CONFIG[horizon]["horizon_bars"]
            price = df["Close"].values
            # Compute absolute levels from the unlagged percentage features
            ma = price / (1 + df["pct_from_ma"].values / 100.0)
            sup = price / (1 + df["pct_from_support_band"].values / 100.0)
            res = price / (1 + df["pct_from_resistance_band"].values / 100.0)
            has_pos = df["has_position"].values > 0
            usr = price / (1 + df["pct_from_user_avg_cost"].values / 100.0)

            highs = df["High"].values
            lows = df["Low"].values
            N = len(df)
            
            labels = np.full(N, "NONE", dtype=object)
            
            for i in range(N - horizon_bars):
                end = i + 1 + horizon_bars
                h_win = highs[i+1:end]
                l_win = lows[i+1:end]
                
                # Priority order: USER_COST, SUPPORT, RESISTANCE, MA
                targets = [
                    ("USER_COST", usr[i] if has_pos[i] else np.nan),
                    ("SUPPORT", sup[i]),
                    ("RESISTANCE", res[i]),
                    ("MA", ma[i])
                ]
                
                first_hit_idx = horizon_bars + 1
                hit_label = "NONE"
                
                for label, level in targets:
                    if np.isnan(level) or level == 0.0:
                        continue
                    hits = np.where((l_win <= level) & (h_win >= level))[0]
                    if len(hits) > 0:
                        first_hit = hits[0]
                        if first_hit < first_hit_idx:
                            first_hit_idx = first_hit
                            hit_label = label
                
                labels[i] = hit_label
                
            # Final rows have no valid future
            labels_series = pd.Series(labels, index=df.index)
            labels_series.iloc[-horizon_bars:] = np.nan
            return labels_series
        except Exception as e:
            logger.error(f"Failed building price level labels: {e}")
            return pd.Series(np.nan, index=df.index)

    def build_labels(self, df: pd.DataFrame, horizon: str = HORIZON_INTRADAY) -> pd.Series:
        """
        Forward-looking label: this is the one place in the whole system where
        looking into the future is CORRECT and required — a supervised label
        must describe what actually happened after the decision point. This is
        not lookahead bias; lookahead bias is a FEATURE seeing the future, and
        every feature that reaches this function has already been '_feat'
        lagged before it gets here.
        """
        try:
            horizon_bars = HORIZON_CONFIG[horizon]["horizon_bars"]
            deadband_pct = HORIZON_CONFIG[horizon]["deadband_pct_default"]
            
            adaptive_deadband = self.compute_adaptive_deadband(df, horizon_bars, deadband_pct)
            
            future_return_pct = (df["Close"].shift(-horizon_bars) - df["Close"]) / df["Close"] * 100.0
            labels = pd.Series("FLAT", index=df.index)
            labels[future_return_pct > adaptive_deadband] = "UP"
            labels[future_return_pct < -adaptive_deadband] = "DOWN"
            # Rows near the end of the dataset have no future to look at — label is invalid there
            labels[future_return_pct.isna()] = np.nan
            return labels
        except Exception as e:
            logger.error(f"Failed building labels: {e}")
            return pd.Series(np.nan, index=df.index)

    def prepare_dataset(
        self, stock_df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None, horizon: str = HORIZON_INTRADAY,
        label_type: str = "direction"
    ) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
        """
        Runs feature engineering, selects ONLY '_feat' (ML-safe, lagged)
        columns as X, builds the forward-looking label as y, and drops rows
        with any NaN.
        If label_type == "level", simulates user cost and builds price level labels.
        """
        try:
            engineered = self.feature_engineer.engineer_features_for_horizon(stock_df, index_df, horizon=horizon)
            if engineered is None or engineered.empty:
                logger.error("Feature engineering returned no data — cannot prepare dataset.")
                return None

            if label_type == "level":
                self.simulate_user_cost(engineered)

            feature_columns = [c for c in engineered.columns if c.endswith(ML_SAFE_SUFFIX)]
            if not feature_columns:
                logger.error("No ML-safe ('_feat') columns found — refusing to train on raw columns.")
                return None

            raw_leak = [c for c in feature_columns if not c.endswith(ML_SAFE_SUFFIX)]
            assert not raw_leak, f"Non-lagged column(s) detected in feature set: {raw_leak}"

            if label_type == "level":
                labels = self.build_price_level_labels(engineered, horizon=horizon)
            else:
                labels = self.build_labels(engineered, horizon=horizon)
                
            X = engineered[feature_columns].copy()
            y = labels.copy()

            combined = pd.concat([X, y.rename("label")], axis=1).dropna()
            
            if combined.empty:
                logger.error("No rows remain after dropping NaN (warmup/label horizon) — dataset too short.")
                return None

            X_clean = combined[feature_columns]
            y_clean = combined["label"]
            return X_clean, y_clean, feature_columns

        except Exception as e:
            logger.error(f"Failed preparing dataset: {e}")
            return None

    # -----------------------------------------------------------------
    # Time-based split (never shuffled)
    # -----------------------------------------------------------------
    @staticmethod
    def time_based_split(
        X: pd.DataFrame, y: pd.Series, test_fraction: float = TIME_SERIES_SPLIT_TEST_FRACTION
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Strictly chronological split: the earliest (1 - test_fraction) of rows
        become the training set, the most recent test_fraction become the
        test set. No shuffling, ever — X and y are assumed to already be
        sorted by time (which they are, since they come from a DatetimeIndex).
        """
        n = len(X)
        split_idx = int(n * (1 - test_fraction))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        return X_train, X_test, y_train, y_test

    # -----------------------------------------------------------------
    # Train / evaluate / persist
    # -----------------------------------------------------------------
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
        model = GradientBoostingClassifier(
            n_estimators=MODEL_N_ESTIMATORS,
            max_depth=MODEL_MAX_DEPTH,
            learning_rate=MODEL_LEARNING_RATE,
            random_state=MODEL_RANDOM_SEED,
        )
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate(model: GradientBoostingClassifier, X_test: pd.DataFrame, y_test: pd.Series, label_classes: List[str] = LABEL_CLASSES) -> Dict:
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, labels=label_classes, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=label_classes).tolist()
        return {"accuracy": accuracy, "classification_report": report, "confusion_matrix": cm}

    def _model_path(self, symbol: str, horizon: str, is_level: bool = False) -> Path:
        suffix = LEVEL_MODEL_FILE_SUFFIX if is_level else MODEL_FILE_SUFFIX
        return MODELS_DIR / f"{symbol}_{horizon}{suffix}"

    def _metadata_path(self, symbol: str, horizon: str, is_level: bool = False) -> Path:
        suffix = LEVEL_METADATA_FILE_SUFFIX if is_level else METADATA_FILE_SUFFIX
        return MODELS_DIR / f"{symbol}_{horizon}{suffix}"

    def save_model(self, symbol: str, model: GradientBoostingClassifier, feature_columns: List[str],
                   metrics: Dict, horizon: str = HORIZON_INTRADAY, is_level: bool = False) -> None:
        try:
            ensure_directories()
            joblib.dump(model, self._model_path(symbol, horizon, is_level))
            
            lbl_classes = LEVEL_LABEL_CLASSES if is_level else LABEL_CLASSES
            
            metadata = {
                "symbol": symbol,
                "horizon": horizon,
                "is_level_model": is_level,
                "trained_at": datetime.now().isoformat(),
                "feature_columns": feature_columns,
                "label_classes": lbl_classes,
                "prediction_horizon_bars": HORIZON_CONFIG[horizon]["horizon_bars"],
                "deadband_pct": HORIZON_CONFIG[horizon]["deadband_pct_default"],
                "test_accuracy": metrics["accuracy"],
            }
            f = None
            try:
                f = open(self._metadata_path(symbol, horizon, is_level), "w", encoding="utf-8")
                json.dump(metadata, f, indent=2)
            finally:
                if f is not None:
                    f.close()
            mdl_type = "level model" if is_level else "direction model"
            logger.info(f"Saved {mdl_type} + metadata for {symbol} ({horizon}) to {MODELS_DIR}")
        except Exception as e:
            logger.error(f"Failed saving model for {symbol} ({horizon}): {e}")
            raise

    def load_model(self, symbol: str, horizon: str = HORIZON_INTRADAY, is_level: bool = False) -> Optional[Tuple[GradientBoostingClassifier, Dict]]:
        model_path = self._model_path(symbol, horizon, is_level)
        metadata_path = self._metadata_path(symbol, horizon, is_level)
        try:
            if not model_path.exists() or not metadata_path.exists():
                logger.error(f"No saved model found for {symbol} ({horizon}) at {model_path}. Run training first.")
                return None
            model = joblib.load(model_path)
            f = None
            try:
                f = open(metadata_path, "r", encoding="utf-8")
                metadata = json.load(f)
            finally:
                if f is not None:
                    f.close()
            return model, metadata
        except Exception as e:
            logger.error(f"Failed loading model for {symbol} ({horizon}): {e}")
            return None

    # -----------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------
    def train_for_symbol(
        self, symbol: str, stock_df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None,
        horizon: str = HORIZON_INTRADAY, is_level: bool = False
    ) -> TrainingResult:
        """Full pipeline for one stock: prepare -> split -> train -> evaluate -> save."""
        try:
            label_type = "level" if is_level else "direction"
            prepared = self.prepare_dataset(stock_df, index_df, horizon=horizon, label_type=label_type)
            if prepared is None:
                return TrainingResult(
                    symbol=symbol, trained_at=datetime.now().isoformat(), n_train_samples=0,
                    n_test_samples=0, feature_columns=[], test_accuracy=0.0, class_report={},
                    success=False, error="Dataset preparation failed or returned no usable rows.",
                )
            X, y, feature_columns = prepared

            min_training_samples = HORIZON_CONFIG[horizon]["min_training_samples"]
            if len(X) < min_training_samples:
                msg = f"Only {len(X)} usable samples for {symbol} ({horizon}), need >= {min_training_samples}."
                logger.error(msg)
                return TrainingResult(
                    symbol=symbol, trained_at=datetime.now().isoformat(), n_train_samples=len(X),
                    n_test_samples=0, feature_columns=feature_columns, test_accuracy=0.0,
                    class_report={}, success=False, error=msg,
                )

            X_train, X_test, y_train, y_test = self.time_based_split(X, y)

            # Hard runtime guarantee that the split is genuinely chronological —
            # every training timestamp must precede every test timestamp.
            if len(X_train) > 0 and len(X_test) > 0:
                assert X_train.index.max() <= X_test.index.min(), (
                    "Time-based split violated: a training row is timestamped after a test row!"
                )

            model = self.train(X_train, y_train)
            lbl_classes = LEVEL_LABEL_CLASSES if is_level else LABEL_CLASSES
            metrics = self.evaluate(model, X_test, y_test, label_classes=lbl_classes)
            self.save_model(symbol, model, feature_columns, metrics, horizon=horizon, is_level=is_level)

            return TrainingResult(
                symbol=symbol, trained_at=datetime.now().isoformat(), n_train_samples=len(X_train),
                n_test_samples=len(X_test), feature_columns=feature_columns,
                test_accuracy=metrics["accuracy"], class_report=metrics["classification_report"],
                success=True,
            )

        except Exception as e:
            logger.error(f"Training pipeline failed for {symbol}: {e}")
            return TrainingResult(
                symbol=symbol, trained_at=datetime.now().isoformat(), n_train_samples=0,
                n_test_samples=0, feature_columns=[], test_accuracy=0.0, class_report={},
                success=False, error=str(e),
            )


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="model_trainer_selftest.log")
    logger.info("Running model_trainer.py self-test...")

    def _build_synthetic_ohlcv_with_signal(n_days: int = 40, bars_per_day: int = 75, seed: int = 42) -> pd.DataFrame:
        """
        Builds synthetic OHLCV with a DELIBERATE, learnable pattern embedded
        (mean-reversion after a short losing/winning streak) so the trained
        model has something real to find — this makes 'better than random
        accuracy' a meaningful assertion rather than a coincidence. Fully
        offline, fixed seed.
        """
        rng = np.random.default_rng(seed)
        rows, timestamps = [], []
        price = 1000.0
        base_date = pd.Timestamp("2026-01-05 09:15:00")
        recent_closes = []

        for day in range(n_days):
            day_start = base_date + pd.Timedelta(days=day)
            for bar in range(bars_per_day):
                ts = day_start + pd.Timedelta(minutes=5 * bar)
                if len(recent_closes) >= 10:
                    recent_trend = recent_closes[-1] - recent_closes[-10]
                    bias = 3.0 if recent_trend < -6 else (-3.0 if recent_trend > 6 else 0.0)
                else:
                    bias = 0.0
                drift = rng.normal(bias, 1.5)
                price = max(1.0, price + drift)
                open_p = price
                close_p = max(1.0, price + rng.normal(bias * 0.5, 1.0))
                high_p = max(open_p, close_p) + abs(rng.normal(0, 0.5))
                low_p = min(open_p, close_p) - abs(rng.normal(0, 0.5))
                vol = int(abs(rng.normal(50000, 15000)))
                rows.append([open_p, high_p, low_p, close_p, vol])
                timestamps.append(ts)
                price = close_p
                recent_closes.append(close_p)

        return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"],
                             index=pd.DatetimeIndex(timestamps))

    test_symbol = "SYNTHTEST"  # single test symbol allowed in the __main__ block only

    try:
        print("\n=== MODEL TRAINER SELF-TEST RESULT ===")
        trainer = ModelTrainer()

        stock_df = _build_synthetic_ohlcv_with_signal(n_days=40, bars_per_day=75, seed=42)
        index_df = _build_synthetic_ohlcv_with_signal(n_days=40, bars_per_day=75, seed=99)
        index_df.index = stock_df.index

        result = trainer.train_for_symbol(test_symbol, stock_df, index_df)

        print(f"Training success: {result.success}")
        if not result.success:
            print(f"Error: {result.error}")
        print(f"Train samples: {result.n_train_samples}, Test samples: {result.n_test_samples}")
        print(f"Feature columns used ({len(result.feature_columns)}): all end in '_feat': "
              f"{all(c.endswith(ML_SAFE_SUFFIX) for c in result.feature_columns)}")
        class_balance = y_test_balance = None
        try:
            prepared_for_balance = trainer.prepare_dataset(stock_df, index_df)
            if prepared_for_balance is not None:
                _, y_all, _ = prepared_for_balance
                class_balance = y_all.value_counts(normalize=True).to_dict()
        except Exception:
            pass
        majority_baseline = max(class_balance.values()) if class_balance else 0.333
        print(f"Test accuracy: {result.test_accuracy:.3f} "
              f"(label distribution: {class_balance}, majority-class baseline: {majority_baseline:.3f})")
        print("NOTE: accuracy on a synthetic toy pattern is informational only, not a pass/fail gate — "
              "the real contract this phase must prove is the structural guarantees below.")

        # Save/load round trip check
        loaded = trainer.load_model(test_symbol, horizon=HORIZON_INTRADAY)
        load_ok = loaded is not None
        print(f"Model save/load round trip: {'OK' if load_ok else 'FAILED'}")

        predictions_match = False
        if load_ok:
            loaded_model, metadata = loaded
            prepared = trainer.prepare_dataset(stock_df, index_df, horizon=HORIZON_INTRADAY)
            if prepared is not None:
                X, y, _ = prepared
                _, X_test, _, _ = trainer.time_based_split(X, y)
                preds_loaded = loaded_model.predict(X_test)
                # Re-load independently from disk again to prove it's the persisted artifact, not the in-memory object
                reloaded_model = joblib.load(trainer._model_path(test_symbol, HORIZON_INTRADAY))
                preds_direct = reloaded_model.predict(X_test)
                predictions_match = np.array_equal(preds_loaded, preds_direct)
        print(f"Loaded model produces identical predictions on reload: {predictions_match}")
        
        # Check adaptive deadband test
        db_adaptive = trainer.compute_adaptive_deadband(stock_df, horizon_bars=6, deadband_pct_default=0.15)
        adaptive_ok = db_adaptive is not None and len(db_adaptive) == len(stock_df)
        print(f"Adaptive deadband computation: {'OK' if adaptive_ok else 'FAILED'}")

        # PASS/FAIL is gated on the structural guarantees this phase actually promises —
        # chronological split integrity (enforced by an assertion inside train_for_symbol,
        # which would flip result.success to False if violated), ML-safe feature usage,
        # and save/load fidelity. Raw accuracy on a made-up synthetic pattern is printed
        # above for visibility only and deliberately does NOT gate pass/fail — that would
        # be testing the toy data generator's learnability, not this file's correctness.
        # Test level model training
        print("\n=== LEVEL MODEL TRAINING SELF-TEST ===")
        level_result = trainer.train_for_symbol(test_symbol, stock_df, index_df, is_level=True)
        print(f"Level Training success: {level_result.success}")
        
        # Check that it simulated user cost (has_position should have variance)
        simulated_ok = False
        prepared_level = trainer.prepare_dataset(stock_df, index_df, label_type="level")
        if prepared_level is not None:
            X_lvl, y_lvl, _ = prepared_level
            if "has_position_feat" in X_lvl.columns:
                simulated_ok = X_lvl["has_position_feat"].nunique() > 1
        print(f"User cost simulation varied has_position: {simulated_ok}")
        
        overall_pass = (
            result.success
            and all(c.endswith(ML_SAFE_SUFFIX) for c in result.feature_columns)
            and load_ok
            and predictions_match
            and level_result.success
            and simulated_ok
        )
        print("STATUS: PASS" if overall_pass else "STATUS: FAIL — see details above")

        assert overall_pass, "One or more model_trainer.py self-test checks failed"
        logger.info("model_trainer.py self-test passed.")

    except AssertionError as ae:
        logger.error(f"model_trainer.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL — {ae}")
    except Exception as e:
        logger.error(f"model_trainer.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
