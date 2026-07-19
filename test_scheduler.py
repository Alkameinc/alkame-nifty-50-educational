import pandas as pd
from scheduler import Scheduler
from config import HORIZON_INTRADAY
import logging
logging.basicConfig(level=logging.DEBUG)

dates = pd.date_range('2026-07-01', periods=100)
stock_df = pd.DataFrame({'Close': [100.0]*100, 'High': [105.0]*100, 'Low': [95.0]*100, 'Open': [100.0]*100, 'Volume': [1000]*100}, index=dates)
index_df = stock_df.copy()

scheduler = Scheduler()
scheduler.data_fetcher.is_market_open = lambda: False
scheduler.get_cached_live_worthiness = lambda s: None

try:
    res = scheduler.run_one_cycle_for_symbol('RELIANCE', stock_df, index_df)
    if res is None:
        print('Returned None! There was a caught exception.')
    else:
        print('Success! No exception caught.')
except Exception as e:
    import traceback
    traceback.print_exc()
