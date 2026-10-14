from typing import Literal, TypedDict


class MarketData(TypedDict):
    symbol: str
    date: str
    open: float
    close: float
    ema20: float
    ema50: float
    ema200: float
    rsi: float
    atr14: float
    gap_pct: float


class MacroContext(TypedDict):
    indicators: dict[str, float]
    errors: dict[str, str]
    enabled_indicators: list[str]
    risk_mode: Literal["RISK_ON", "CAUTIOUS", "RISK_OFF", "UNAVAILABLE"]


class QuantData(TypedDict):
    long_score: int
    short_score: int
    bias: Literal["LONG", "SHORT", "NEUTRAL"]


class RiskData(TypedDict):
    review_candidate: bool
    reason: str


class NewsArticle(TypedDict):
    source: str
    title: str
    summary: str
    url: str
    published_at: str
    sentiment: Literal["POSITIVE", "NEGATIVE", "MIXED", "UNCLEAR"]
    sentiment_source: str
    market_impact: Literal["RISK_ON", "RISK_OFF", "MIXED", "UNCLEAR"]
    impact_reason: str
    topics: list[str]


class NewsContext(TypedDict):
    active: bool
    articles: list[NewsArticle]
    errors: dict[str, str]
    enabled_sources: list[str]
    sentiment_method: str
    overall_sentiment: Literal["POSITIVE", "NEGATIVE", "MIXED", "UNCLEAR"]


class TradingState(TypedDict, total=False):
    market_data: MarketData
    macro_context: MacroContext
    quant: QuantData
    risk: RiskData
    news_context: NewsContext
    report: str
    llm_summary: str
