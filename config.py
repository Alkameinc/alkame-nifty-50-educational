# 1. Standard library imports
import os
import logging
from pathlib import Path
from datetime import time as dt_time

# 2. Third-party imports
# (none required for config itself; kept dependency-free on purpose)

# 3. Local imports
# (this is the base file — nothing to import)

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. CONSTANTS
# ---------------------------------------------------------------------------

# --- Project root / directory layout -----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DB_DIR = PROJECT_ROOT / "db"
DB_PATH = DB_DIR / "predictor.sqlite3"
MACRO_CALENDAR_PATH = DATA_DIR / "macro_calendar.csv"
SECTOR_MAP_PATH = DATA_DIR / "sector_map.csv"

REQUIRED_DIRS = [DATA_DIR, CACHE_DIR, MODELS_DIR, LOGS_DIR, DB_DIR]


def ensure_directories() -> None:
    """Create all required project directories if they do not already exist."""
    for directory in REQUIRED_DIRS:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise


# --- Logging setup helper (used by every other module) ------------------
def configure_logging(log_filename: str = "app.log", level: int = logging.INFO) -> None:
    """Configure root logging to write to both console and a rotating file."""
    try:
        ensure_directories()
        log_path = LOGS_DIR / log_filename
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
    except Exception as e:
        # Fall back to console-only logging so the app doesn't die on a logging bug
        logging.basicConfig(level=level)
        logger.error(f"Failed to configure file logging, falling back to console only: {e}")


# --- API keys / secrets (never hardcode actual key values) --------------
MARKETAUX_API_KEY = os.environ.get("MARKETAUX_API_KEY", "")
IMD_API_KEY = os.environ.get("IMD_API_KEY", "")

REQUIRED_ENV_VARS = ["MARKETAUX_API_KEY"]   # IMD key is optional (monsoon feature degrades gracefully without it)


# --- NIFTY 50 universe ----------------------------------------------------
# NOTE: NSE Indices reviews NIFTY 50 composition semi-annually (March & September).
# This list must be verified/updated after each review. Source of truth:
# https://www.nseindia.com/products-services/indices-nifty50-index
NIFTY50_SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "BHARTIARTL", "SBIN",
    "LT", "ITC", "HINDUNILVR", "BAJFINANCE", "KOTAKBANK", "AXISBANK", "MARUTI",
    "SUNPHARMA", "TITAN", "ULTRACEMCO", "NTPC", "HCLTECH", "ONGC", "ADANIENT",
    "ADANIPORTS", "M&M", "COALINDIA", "ASIANPAINT", "BAJAJFINSV", "WIPRO",
    "NESTLEIND", "POWERGRID", "JSWSTEEL", "TATASTEEL", "GRASIM",
    "TECHM", "HINDALCO", "CIPLA", "DRREDDY", "EICHERMOT", "BRITANNIA",
    "DIVISLAB", "BPCL", "HEROMOTOCO", "APOLLOHOSP", "SBILIFE", "HDFCLIFE",
    "INDUSINDBK", "BAJAJ-AUTO", "UPL", "SHRIRAMFIN", "TATACONSUM",
]

# yfinance requires the ".NS" suffix for NSE-listed equities.
YFINANCE_SUFFIX = ".NS"


def to_yfinance_ticker(nse_symbol: str) -> str:
    """Convert a bare NSE symbol (e.g. 'RELIANCE') to a yfinance ticker (e.g. 'RELIANCE.NS')."""
    return f"{nse_symbol}{YFINANCE_SUFFIX}"


NIFTY50_YFINANCE_TICKERS = [to_yfinance_ticker(s) for s in NIFTY50_SYMBOLS]

NIFTY_INDEX_TICKER = "^NSEI"  # NIFTY 50 index itself, used as the baseline for edge checks

# --- Sector mapping (used by event_classifier.py for SECTOR-scope tagging) ---
# Kept intentionally simple/broad; refine over time as real classification needs emerge.
SECTOR_MAP = {
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", "COALINDIA": "Energy",
    "NTPC": "Power", "POWERGRID": "Power", "ADANIENT": "Conglomerate", "ADANIPORTS": "Infra",
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking", "KOTAKBANK": "Banking",
    "AXISBANK": "Banking", "INDUSINDBK": "Banking",
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC", "SHRIRAMFIN": "NBFC",
    "SBILIFE": "Insurance", "HDFCLIFE": "Insurance",
    "INFY": "IT", "TCS": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "BHARTIARTL": "Telecom",
    "LT": "Infra", "ULTRACEMCO": "Cement", "GRASIM": "Cement",
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG", "TATACONSUM": "FMCG",
    "MARUTI": "Auto", "M&M": "Auto", "EICHERMOT": "Auto",
    "HEROMOTOCO": "Auto", "BAJAJ-AUTO": "Auto",
    "SUNPHARMA": "Pharma", "CIPLA": "Pharma", "DRREDDY": "Pharma", "DIVISLAB": "Pharma",
    "APOLLOHOSP": "Healthcare",
    "TITAN": "ConsumerDurables", "ASIANPAINT": "ConsumerDurables",
    "JSWSTEEL": "Metals", "TATASTEEL": "Metals", "HINDALCO": "Metals",
    "UPL": "Agrochemicals",
}

# Sectors considered exposed to specific macro/global-risk triggers.
# Used by global_risk_monitor.py and event_classifier.py to scope MARKET-level events
# down to the sectors actually affected, instead of blanket-tagging all 50 stocks.
CRUDE_SENSITIVE_SECTORS = ["Energy", "Auto"]           # importers hurt, oil producers mixed
INR_WEAKNESS_BENEFICIARY_SECTORS = ["IT", "Pharma"]     # exporters benefit from weak INR
INR_WEAKNESS_HURT_SECTORS = ["Energy", "Auto"]          # importers hurt by weak INR
MONSOON_SENSITIVE_SECTORS = ["FMCG", "Auto", "Agrochemicals"]
RATE_SENSITIVE_SECTORS = ["Banking", "NBFC", "Insurance", "ConsumerDurables", "Auto"]

# --- Global / cross-asset tickers (yfinance) ------------------------------
GLOBAL_TICKERS = {
    "DOLLAR_INDEX": "DX-Y.NYB",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "CRUDE_BRENT": "BZ=F",
    "CRUDE_WTI": "CL=F",
    "USD_INR": "INR=X",
    "US_SP500": "^GSPC",
    "US_NASDAQ": "^IXIC",
    "VIX": "^VIX",
}

# --- Market session (IST) -------------------------------------------------
MARKET_OPEN_TIME = dt_time(9, 15)
MARKET_CLOSE_TIME = dt_time(15, 30)
PRE_MARKET_OPEN_TIME = dt_time(9, 0)
MARKET_TIMEZONE = "Asia/Kolkata"

# --- Data fetch parameters -------------------------------------------------
BAR_INTERVAL = "5m"          # 5-minute bars: ~60-day history, stable. Change to "1m" for finer/shorter history.
BAR_HISTORY_PERIOD = "60d"
DATA_STALENESS_THRESHOLD_MINUTES = 15   # if last bar older than this during market hours -> flag UNSAFE
DATA_STALENESS_THRESHOLD_TRADING_DAYS = 1  # if last daily bar is more than 1 trading day old -> flag UNSAFE

# --- Technical event thresholds --------------------------------------------
ORB_MINUTES = 15                    # opening range breakout window
GAP_THRESHOLD_PCT = 1.0             # % gap vs prior close to flag a gap event
VOLUME_SPIKE_MULTIPLIER = 2.5       # current bar volume vs rolling average to flag spike
VOLUME_SPIKE_LOOKBACK_BARS = 20
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD_DEV = 2.0
ATR_PERIOD = 14
ATR_EXPANSION_MULTIPLIER = 1.8      # current ATR vs its own rolling average to flag volatility expansion
MA_FAST_PERIOD = 9
MA_SLOW_PERIOD = 21
LOW_LIQUIDITY_VOLUME_FLOOR = 10000  # rolling average volume below this -> confidence downgrade

# --- Relative / market-context thresholds -----------------------------------
OUTPERFORMANCE_THRESHOLD_PCT = 1.5   # stock % move minus index % move beyond this -> flag
CORRELATION_LOOKBACK_BARS = 60
CORRELATION_BREAKDOWN_THRESHOLD = 0.3  # rolling correlation with NIFTY drops below this -> flag

# --- Global risk monitor thresholds ------------------------------------------
GLOBAL_RISK_ZSCORE_WARN_THRESHOLD = 1.5     # composite z-score above this -> show banner
GLOBAL_RISK_ZSCORE_CRISIS_THRESHOLD = 2.5   # above this -> stronger banner language
GLOBAL_RISK_CONFIDENCE_DOWNGRADE_ELEVATED = 0.85   # multiplier applied to confidence when toggle ON, elevated
GLOBAL_RISK_CONFIDENCE_DOWNGRADE_CRISIS = 0.60     # multiplier applied to confidence when toggle ON, crisis
GLOBAL_RISK_LOOKBACK_DAYS = 20               # window used to compute rolling mean/std for z-score

# --- News / sentiment parameters ---------------------------------------------
MARKETAUX_BASE_URL = "https://api.marketaux.com/v1/news/all"
MARKETAUX_COUNTRY = "in"
NEWS_FETCH_LIMIT = 20
NEWS_STALENESS_HOURS = 24            # news older than this is not considered "active" for signal purposes
GOOGLE_NEWS_RSS_BASE = "https://news.google.com/rss/search"

# --- NSE corporate events fetch parameters -----------------------------------
NSE_RATE_LIMIT_DELAY_SECONDS = 0.4   # NSE throttles to ~3 req/sec; we stay comfortably under that
NSE_ANNOUNCEMENT_REFRESH_MINUTES = 10  # corporate announcements refresh cadence during market hours

# --- Model / validation parameters -------------------------------------------
TIME_SERIES_SPLIT_TEST_FRACTION = 0.2   # always time-based, never random shuffle
MIN_CALIBRATION_SAMPLES = 50            # minimum historical predictions needed before trusting confidence scores
CALIBRATION_N_BINS = 10                 # number of confidence buckets used to build the reliability curve
CALIBRATION_ECE_THRESHOLD = 0.10        # max acceptable Expected Calibration Error to call confidence "well calibrated"
EDGE_CHECK_MIN_ALPHA_PCT = 0.0          # signal must show >0 edge vs NIFTY baseline in backtest to be enabled live

# --- Multi-Horizon parameters (Phase 2) ---------------------------------------
HORIZON_INTRADAY = "INTRADAY"
HORIZON_3D = "3D"
HORIZON_7D = "7D"
HORIZON_30D = "30D"
HORIZON_3M = "3M"
HORIZON_6M = "6M"
HORIZON_1Y = "1Y"

ALL_HORIZONS = [HORIZON_INTRADAY, HORIZON_3D, HORIZON_7D, HORIZON_30D, HORIZON_3M, HORIZON_6M, HORIZON_1Y]

HORIZON_CONFIG = {
    HORIZON_INTRADAY: {
        "bar_interval": "5m", "horizon_bars": 6, "deadband_pct_default": 0.15,
        "history_period": "60d", "min_training_samples": 500,
        "retrain_cadence_days": 1,
    },
    HORIZON_3D: {
        "bar_interval": "1d", "horizon_bars": 3, "deadband_pct_default": 1.0,
        "history_period": "2y", "min_training_samples": 300,
        "retrain_cadence_days": 7,
    },
    HORIZON_7D: {
        "bar_interval": "1d", "horizon_bars": 5, "deadband_pct_default": 1.5,
        "history_period": "3y", "min_training_samples": 400,
        "retrain_cadence_days": 7,
    },
    HORIZON_30D: {
        "bar_interval": "1d", "horizon_bars": 21, "deadband_pct_default": 3.0,
        "history_period": "5y", "min_training_samples": 500,
        "retrain_cadence_days": 14,
    },
    HORIZON_3M: {
        "bar_interval": "1d", "horizon_bars": 63, "deadband_pct_default": 5.0,
        "history_period": "7y", "min_training_samples": 500,
        "retrain_cadence_days": 30,
    },
    HORIZON_6M: {
        "bar_interval": "1d", "horizon_bars": 126, "deadband_pct_default": 8.0,
        "history_period": "10y", "min_training_samples": 400,
        "retrain_cadence_days": 60,
    },
    HORIZON_1Y: {
        "bar_interval": "1d", "horizon_bars": 252, "deadband_pct_default": 12.0,
        "history_period": "max", "min_training_samples": 300,
        "retrain_cadence_days": 90,
    },
}

EVENT_IMPACT_HORIZON = {
    "RBI_POLICY": HORIZON_30D,
    "GDP_RELEASE": HORIZON_3M,
    "UNION_BUDGET": HORIZON_1Y,
    "ELECTION": HORIZON_1Y,
    "FESTIVE_WINDOW": HORIZON_3M,
    "MONSOON_STATUS": HORIZON_6M,
    "FDI_FLOW_RELEASE": HORIZON_6M,
    "GEOPOLITICAL": HORIZON_30D,
    "QUARTERLY_EARNINGS": HORIZON_30D,
    "REGULATORY_CHANGE": HORIZON_6M,
    "CORPORATE_ANNOUNCEMENT": HORIZON_7D,
    "NEWS_HEADLINE": HORIZON_3D,
    "MACRO_OTHER": HORIZON_30D,
    "CLASSIFICATION_ERROR": HORIZON_INTRADAY,
}

HORIZON_TO_MA_PERIOD = {
    HORIZON_INTRADAY: 9,
    HORIZON_3D: 9,
    HORIZON_7D: 9,
    HORIZON_30D: 50,
    HORIZON_3M: 50,
    HORIZON_6M: 200,
    HORIZON_1Y: 200,
}

HORIZON_TO_SR_METHOD = {
    HORIZON_INTRADAY: "bollinger",
    HORIZON_3D: "bollinger",
    HORIZON_7D: "bollinger",
    HORIZON_30D: "swing_levels",
    HORIZON_3M: "swing_levels",
    HORIZON_6M: "swing_levels",
    HORIZON_1Y: "swing_levels",
}

# --- Model training parameters (Phase 9) --------------------------------------
PREDICTION_HORIZON_BARS = HORIZON_CONFIG[HORIZON_INTRADAY]["horizon_bars"]
PREDICTION_DEADBAND_PCT = HORIZON_CONFIG[HORIZON_INTRADAY]["deadband_pct_default"]
LABEL_CLASSES = ["DOWN", "FLAT", "UP"]
MODEL_RANDOM_SEED = 42
MODEL_N_ESTIMATORS = 200
MODEL_MAX_DEPTH = 4
MODEL_LEARNING_RATE = 0.05
MIN_TRAINING_SAMPLES_PER_STOCK = 500    # below this, we refuse to train — too little data to trust

# --- Portfolio / Position Sizing (Phase 11) -----------------------------------
PORTFOLIO_TOTAL_CAPITAL = 10_00_000   # Default 10 Lakhs INR
MAX_POSITION_SIZE_PCT = 0.10          # Max 10% of portfolio per stock
MAX_DCA_STEPS = 4                     # Max number of tranches in a DCA ladder

# --- Ensemble parameters (Phase 10) --------------------------------------------
ENSEMBLE_MODEL_TYPES = ["gradient_boosting", "random_forest", "logistic_regression"]
ENSEMBLE_RF_N_ESTIMATORS = 200
ENSEMBLE_RF_MAX_DEPTH = 6
ENSEMBLE_LR_MAX_ITER = 500

# --- Backtest cost assumptions ------------------------------------------------
SLIPPAGE_BPS = 5          # 0.05% slippage per trade
TRANSACTION_COST_BPS = 3  # 0.03% brokerage + STT + other charges approximation

# --- Scheduler parameters (Phase 16) -------------------------------------------
SCHEDULER_INTERVAL_MINUTES = 5          # how often the live pipeline cycles during market hours (matches BAR_INTERVAL)
LIVE_WORTHINESS_REFRESH_HOURS = 24      # how often the backtest-derived live/edge status is refreshed per symbol

# --- Health Monitor parameters (Phase 1) -------------------------------------------
HEALTH_DEGRADED_THRESHOLD = 2
HEALTH_DOWN_THRESHOLD = 5
HEALTH_THRESHOLD_OVERRIDES = {
    "data_fetcher": {"degraded": 1, "down": 3}
}

# ---------------------------------------------------------------------------
# 6. Functions
# ---------------------------------------------------------------------------

def validate_config() -> list:
    """
    Run basic sanity checks on the configuration. Returns a list of warning
    strings (empty list means everything looks fine). Does not raise, so
    callers can decide how strictly to enforce these checks.
    """
    warnings = []
    try:
        if len(NIFTY50_SYMBOLS) != 50:
            warnings.append(
                f"NIFTY50_SYMBOLS has {len(NIFTY50_SYMBOLS)} entries, expected 50. "
                "Index composition may have changed — verify against nseindia.com."
            )

        unmapped = [s for s in NIFTY50_SYMBOLS if s not in SECTOR_MAP]
        if unmapped:
            warnings.append(f"{len(unmapped)} symbols missing from SECTOR_MAP: {unmapped}")

        for var_name in REQUIRED_ENV_VARS:
            if not os.environ.get(var_name):
                warnings.append(
                    f"Environment variable '{var_name}' is not set. "
                    "Related features will be degraded or disabled."
                )

        if not IMD_API_KEY:
            warnings.append(
                "IMD_API_KEY not set — monsoon data will fall back to manual entry only."
            )

        for horizon in ALL_HORIZONS:
            if horizon not in HORIZON_CONFIG:
                warnings.append(f"HORIZON_CONFIG missing entry for {horizon}")
            else:
                expected_keys = {"bar_interval", "horizon_bars", "deadband_pct_default", "history_period", "min_training_samples", "retrain_cadence_days"}
                missing_keys = expected_keys - set(HORIZON_CONFIG[horizon].keys())
                if missing_keys:
                    warnings.append(f"HORIZON_CONFIG[{horizon}] is missing keys: {missing_keys}")

    except Exception as e:
        logger.error(f"Error while validating config: {e}")
        warnings.append(f"validate_config() raised an exception: {e}")

    return warnings


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="config_selftest.log")
    logger.info("Running config.py self-test...")

    try:
        ensure_directories()
        logger.info(f"Project root: {PROJECT_ROOT}")
        logger.info(f"Directories ensured: {[str(d) for d in REQUIRED_DIRS]}")

        logger.info(f"NIFTY50 symbol count: {len(NIFTY50_SYMBOLS)}")
        logger.info(f"Sample yfinance tickers: {NIFTY50_YFINANCE_TICKERS[:3]}")
        logger.info(f"Global tickers configured: {list(GLOBAL_TICKERS.keys())}")
        logger.info(f"Horizons configured: {len(ALL_HORIZONS)}")

        warnings = validate_config()
        if warnings:
            logger.warning(f"Config validation produced {len(warnings)} warning(s):")
            for w in warnings:
                logger.warning(f"  - {w}")
        else:
            logger.info("Config validation passed with no warnings.")

        print("\n=== CONFIG SELF-TEST RESULT ===")
        print(f"NIFTY50 symbols: {len(NIFTY50_SYMBOLS)}")
        print(f"Sector map coverage: {len(SECTOR_MAP)}/{len(NIFTY50_SYMBOLS)}")
        print(f"Global tickers: {len(GLOBAL_TICKERS)}")
        print(f"Warnings: {len(warnings)}")
        print("STATUS: PASS" if len(NIFTY50_SYMBOLS) == 50 else "STATUS: CHECK WARNINGS ABOVE")

    except Exception as e:
        logger.error(f"config.py self-test failed: {e}")
        print(f"STATUS: FAIL — {e}")
