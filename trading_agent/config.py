import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class OllamaConfig:
    active: bool
    base_url: str
    model: str


@dataclass(frozen=True)
class MacroIndicatorConfig:
    name: str
    ticker: str
    active: bool


@dataclass(frozen=True)
class NewsFeedConfig:
    name: str
    url: str
    active: bool


@dataclass(frozen=True)
class NewsConfig:
    active: bool
    lookback_hours: int
    max_articles: int
    gdelt_active: bool
    gdelt_query: str
    rss_feeds: tuple[NewsFeedConfig, ...]


@dataclass(frozen=True)
class AppConfig:
    ollama: OllamaConfig
    macro_indicators: tuple[MacroIndicatorConfig, ...]
    news: NewsConfig | None = None


def load_config(config_path: Path | None = None) -> AppConfig:
    path = config_path or Path(__file__).resolve().parent.parent / "config.json"
    with path.open(encoding="utf-8") as config_file:
        raw_config = json.load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError(f"Configuration in {path} must be a JSON object.")

    raw_ollama = raw_config.get("ollama")
    if not isinstance(raw_ollama, dict):
        raise ValueError(f"Configuration in {path} must contain an 'ollama' object.")
    active = raw_ollama.get("active", False)
    base_url = raw_ollama.get("base_url", "")
    model = raw_ollama.get("model", "llama3.3")
    if not isinstance(active, bool):
        raise ValueError("'ollama.active' must be true or false.")
    if not isinstance(base_url, str):
        raise ValueError("'ollama.base_url' must be a string.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("'ollama.model' must be a non-empty string.")

    base_url = base_url.strip().rstrip("/")
    if active and not base_url:
        raise ValueError("'ollama.base_url' is required when Ollama is active.")
    if base_url:
        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("'ollama.base_url' must be an HTTP(S) URL.")
    ollama_config = OllamaConfig(active, base_url, model.strip())

    raw_news = raw_config.get("news", {})
    if not isinstance(raw_news, dict):
        raise ValueError("'news' must be a JSON object.")
    news_active = raw_news.get("active", False)
    lookback_hours = raw_news.get("lookback_hours", 24)
    max_articles = raw_news.get("max_articles", 10)
    if not isinstance(news_active, bool):
        raise ValueError("'news.active' must be true or false.")
    if (
        not isinstance(lookback_hours, int)
        or isinstance(lookback_hours, bool)
        or not 1 <= lookback_hours <= 168
    ):
        raise ValueError("'news.lookback_hours' must be an integer from 1 to 168.")
    if (
        not isinstance(max_articles, int)
        or isinstance(max_articles, bool)
        or not 1 <= max_articles <= 100
    ):
        raise ValueError("'news.max_articles' must be an integer from 1 to 100.")

    raw_gdelt = raw_news.get("gdelt", {})
    if not isinstance(raw_gdelt, dict):
        raise ValueError("'news.gdelt' must be a JSON object.")
    gdelt_active = raw_gdelt.get("active", True)
    gdelt_query = raw_gdelt.get(
        "query",
        "Trump (market OR stocks OR tariff OR trade OR sanctions OR oil OR rates OR Fed)",
    )
    if not isinstance(gdelt_active, bool):
        raise ValueError("'news.gdelt.active' must be true or false.")
    if not isinstance(gdelt_query, str) or not gdelt_query.strip():
        raise ValueError("'news.gdelt.query' must be a non-empty string.")

    raw_feeds = raw_news.get("rss_feeds", [])
    if not isinstance(raw_feeds, list):
        raise ValueError("'news.rss_feeds' must be a JSON array.")
    feeds: list[NewsFeedConfig] = []
    for index, raw_feed in enumerate(raw_feeds):
        if not isinstance(raw_feed, dict):
            raise ValueError(f"'news.rss_feeds[{index}]' must be a JSON object.")
        name = raw_feed.get("name")
        url = raw_feed.get("url")
        feed_active = raw_feed.get("active", True)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"'news.rss_feeds[{index}].name' must be a non-empty string.")
        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"'news.rss_feeds[{index}].url' must be a non-empty string.")
        if not isinstance(feed_active, bool):
            raise ValueError(f"'news.rss_feeds[{index}].active' must be true or false.")
        parsed_url = urlparse(url.strip())
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError(f"'news.rss_feeds[{index}].url' must be an HTTP(S) URL.")
        feeds.append(NewsFeedConfig(name.strip(), url.strip(), feed_active))
    news_config = NewsConfig(
        active=news_active,
        lookback_hours=lookback_hours,
        max_articles=max_articles,
        gdelt_active=gdelt_active,
        gdelt_query=gdelt_query.strip(),
        rss_feeds=tuple(feeds),
    )

    raw_indicators = raw_config.get("macro_indicators", {})
    if not isinstance(raw_indicators, dict):
        raise ValueError("'macro_indicators' must be a JSON object.")
    indicators: list[MacroIndicatorConfig] = []
    for name, raw_indicator in raw_indicators.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Macro indicator names must be non-empty strings.")
        if not isinstance(raw_indicator, dict):
            raise ValueError(f"Macro indicator '{name}' must be a JSON object.")
        ticker = raw_indicator.get("ticker")
        indicator_active = raw_indicator.get("active", True)
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError(f"Macro indicator '{name}' needs a non-empty ticker.")
        if not isinstance(indicator_active, bool):
            raise ValueError(f"Macro indicator '{name}'.active must be true or false.")
        indicators.append(
            MacroIndicatorConfig(
                name=name.strip(),
                ticker=ticker.strip(),
                active=indicator_active,
            )
        )

    return AppConfig(
        ollama=ollama_config,
        macro_indicators=tuple(indicators),
        news=news_config,
    )
