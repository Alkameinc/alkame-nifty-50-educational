from functools import lru_cache

from predictor import Predictor
from history_manager import HistoryManager
from human_insight_manager import HumanInsightManager
from data_fetcher import DataFetcher


@lru_cache()
def get_data_fetcher() -> DataFetcher:
    return DataFetcher()


@lru_cache()
def get_predictor() -> Predictor:
    return Predictor()


@lru_cache()
def get_history_manager() -> HistoryManager:
    return HistoryManager()


@lru_cache()
def get_insight_manager() -> HumanInsightManager:
    return HumanInsightManager()