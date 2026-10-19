import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trading_agent import agents, config as config_module, graph, market_data, news
from trading_agent.analysis import quant_agent
from trading_agent.agents import report_agent, risk_agent


def test_config(
    *,
    ollama_active: bool = False,
) -> config_module.AppConfig:
    return config_module.AppConfig(
        ollama=config_module.OllamaConfig(
            active=ollama_active,
            base_url="http://ollama.local:11434",
            model="qwen3:14b",
        ),
        macro_indicators=(
            config_module.MacroIndicatorConfig(
                name="vix",
                ticker="^VIX",
                active=True,
            ),
        ),
        news=None,
    )


class QuantAgentTests(unittest.TestCase):
    def test_bullish_setup_gets_long_bias(self) -> None:
        result = quant_agent(
            {
                "market_data": {
                    "symbol": "^GDAXI",
                    "date": "2026-10-09",
                    "open": 159.0,
                    "close": 160.0,
                    "ema20": 155.0,
                    "ema50": 150.0,
                    "ema200": 145.0,
                    "rsi": 60.0,
                    "atr14": 2.0,
                    "gap_pct": 0.5,
                }
            }
        )
        self.assertEqual(result["quant"]["bias"], "LONG")
        self.assertEqual(result["quant"]["long_score"], 100)

    def test_bearish_setup_gets_short_bias(self) -> None:
        result = quant_agent(
            {
                "market_data": {
                    "date": "2026-10-09",
                    "close": 140.0,
                    "ema20": 145.0,
                    "ema50": 150.0,
                    "rsi": 40.0,
                }
            }
        )
        self.assertEqual(result["quant"]["bias"], "SHORT")
        self.assertEqual(result["quant"]["short_score"], 100)

    def test_mixed_setup_is_neutral(self) -> None:
        result = quant_agent(
            {
                "market_data": {
                    "date": "2026-10-09",
                    "close": 160.0,
                    "ema20": 155.0,
                    "ema50": 158.0,
                    "rsi": 75.0,
                }
            }
        )
        self.assertEqual(result["quant"]["bias"], "NEUTRAL")


class RiskAgentTests(unittest.TestCase):
    def test_high_vix_blocks_review_candidate(self) -> None:
        result = risk_agent(
            {
                "quant": {"long_score": 100, "short_score": 0, "bias": "LONG"},
                "macro_context": {
                    "indicators": {"vix": 30.0},
                    "errors": {},
                    "enabled_indicators": ["vix"],
                    "risk_mode": "RISK_OFF",
                },
            }
        )
        self.assertFalse(result["risk"]["review_candidate"])

    def test_elevated_vix_keeps_candidate_with_warning(self) -> None:
        result = risk_agent(
            {
                "quant": {"long_score": 100, "short_score": 0, "bias": "LONG"},
                "macro_context": {
                    "indicators": {"vix": 25.0},
                    "errors": {},
                    "enabled_indicators": ["vix"],
                    "risk_mode": "CAUTIOUS",
                },
            }
        )
        self.assertTrue(result["risk"]["review_candidate"])
        self.assertIn("manual review advised", result["risk"]["reason"])

    def test_missing_enabled_vix_blocks_manual_review_candidate(self) -> None:
        result = risk_agent(
            {
                "quant": {"long_score": 100, "short_score": 0, "bias": "LONG"},
                "macro_context": {
                    "indicators": {},
                    "errors": {"vix": "feed unavailable"},
                    "enabled_indicators": ["vix"],
                    "risk_mode": "UNAVAILABLE",
                },
            }
        )
        self.assertFalse(result["risk"]["review_candidate"])
        self.assertIn("VIX is configured but unavailable", result["risk"]["reason"])


class ReportAgentTests(unittest.TestCase):
    def test_report_is_deterministic_and_disclaims_automatic_trading(self) -> None:
        result = report_agent(
            {
                "market_data": {
                    "symbol": "^GDAXI",
                    "date": "2026-10-09",
                    "open": 159.0,
                    "close": 160.0,
                    "ema20": 155.0,
                    "ema50": 150.0,
                    "ema200": 145.0,
                    "rsi": 60.0,
                    "atr14": 2.0,
                    "gap_pct": 0.5,
                },
                "macro_context": {
                    "indicators": {"vix": 16.0},
                    "errors": {},
                    "enabled_indicators": ["vix"],
                    "risk_mode": "RISK_ON",
                },
                "quant": {"long_score": 100, "short_score": 0, "bias": "LONG"},
                "risk": {
                    "review_candidate": True,
                    "reason": "Signal passed basic MVP filters; manual review required.",
                },
            }
        )
        self.assertIn("Manual-review candidate: True", result["report"])
        self.assertIn("EMA200: 145.00", result["report"])
        self.assertIn("ATR14: 2.00", result["report"])
        self.assertIn("Gap from previous close: 0.50%", result["report"])
        self.assertIn("not a validated", result["report"])
        self.assertNotIn("Trade allowed:", result["report"])

    def test_report_formats_news_and_avoids_repeating_rss_provider_as_source(self) -> None:
        result = report_agent(
            {
                "market_data": {
                    "symbol": "^GDAXI",
                    "date": "2026-10-15",
                    "open": 159.0,
                    "close": 160.0,
                    "ema20": 155.0,
                    "ema50": 150.0,
                    "ema200": 145.0,
                    "rsi": 60.0,
                    "atr14": 2.0,
                    "gap_pct": 0.5,
                },
                "macro_context": {
                    "indicators": {},
                    "errors": {},
                    "enabled_indicators": [],
                    "risk_mode": "UNAVAILABLE",
                },
                "quant": {"long_score": 100, "short_score": 0, "bias": "LONG"},
                "risk": {"review_candidate": True, "reason": "Manual review required."},
                "news_context": {
                    "active": True,
                    "articles": [
                        {
                            "provider": "CNBC US Top News",
                            "source": "CNBC US Top News",
                            "title": "Trump tariff news moves markets",
                            "summary": "",
                            "url": "https://cnbc.com/story",
                            "published_at": "2026-10-15T19:43+00:00",
                            "sentiment": "NEGATIVE",
                            "sentiment_source": "keyword heuristic",
                            "market_impact": "RISK_OFF",
                            "impact_reason": "Tariff uncertainty may increase market risk.",
                            "topics": ["trade_policy"],
                        },
                        {
                            "provider": "GDELT",
                            "source": "example.com",
                            "title": "Trump market headline from GDELT",
                            "summary": "",
                            "url": "https://example.com/story",
                            "published_at": "2026-10-15T18:43+00:00",
                            "sentiment": "UNCLEAR",
                            "sentiment_source": "keyword heuristic",
                            "market_impact": "UNCLEAR",
                            "impact_reason": "No clear market impact.",
                            "topics": [],
                        },
                    ],
                    "errors": {},
                    "enabled_sources": ["GDELT", "CNBC US Top News"],
                    "sentiment_method": "keyword heuristic",
                    "overall_sentiment": "MIXED",
                },
            }
        )["report"]

        self.assertIn(
            "- [2026-10-15T19:43+00:00] Provider: CNBC US Top News\n"
            "  News: Trump tariff news moves markets\n"
            "  Sentiment: NEGATIVE\n"
            "  Potential market impact: RISK_OFF\n"
            "  Topics: trade_policy\n"
            "  Assessment: Tariff uncertainty may increase market risk.\n"
            "  URL: https://cnbc.com/story",
            result,
        )
        self.assertIn(
            "- [2026-10-15T18:43+00:00] Provider: GDELT | Source: example.com\n"
            "  News: Trump market headline from GDELT\n"
            "  Sentiment: UNCLEAR\n"
            "  Potential market impact: UNCLEAR\n"
            "  Topics: market policy\n"
            "  Assessment: No clear market impact.\n"
            "  URL: https://example.com/story",
            result,
        )
        self.assertNotIn("Sources: GDELT, CNBC US Top News", result)


class GraphIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        close = pd.Series(
            [15000.0 + i * 5 for i in range(250)],
            index=pd.date_range("2025-10-01", periods=250, freq="B"),
        )
        self.dax = pd.DataFrame(
            {
                "Open": close - 1,
                "High": close + 5,
                "Low": close - 5,
                "Close": close,
            },
            index=close.index,
        )
        vix_close = pd.Series(
            [16.0, 15.5, 16.2, 15.8, 16.1],
            index=pd.date_range("2026-10-05", periods=5, freq="B"),
        )
        self.vix = pd.DataFrame(
            {
                "Open": vix_close - 0.1,
                "High": vix_close + 0.5,
                "Low": vix_close - 0.5,
                "Close": vix_close,
            },
            index=vix_close.index,
        )

    def _download(self, ticker: str, _: str) -> pd.DataFrame:
        return self.dax if ticker in {"^GDAXI", "^ATX"} else self.vix

    def test_graph_runs_without_ollama(self) -> None:
        with (
            patch.object(market_data, "download_ohlc", side_effect=self._download),
        ):
            result = graph.build_graph("^ATX", test_config()).invoke({})

        self.assertIn("Symbol: ^ATX", result["report"])
        self.assertNotIn("llm_summary", result)
        market = result["market_data"]
        self.assertGreater(market["ema200"], 0)
        self.assertGreater(market["atr14"], 0)
        expected_gap = (
            self.dax["Open"].iloc[-1] / self.dax["Close"].iloc[-2] - 1
        ) * 100
        self.assertAlmostEqual(market["gap_pct"], expected_gap)

    def test_macro_context_uses_all_active_configured_symbols(self) -> None:
        configured_indicators = (
            config_module.MacroIndicatorConfig("vix", "^VIX", True),
            config_module.MacroIndicatorConfig("eur_usd", "EURUSD=X", True),
            config_module.MacroIndicatorConfig("us10y", "^TNX", True),
            config_module.MacroIndicatorConfig("nasdaq_futures", "NQ=F", True),
            config_module.MacroIndicatorConfig("sp500_futures", "ES=F", True),
        )
        latest_values = {
            "^VIX": 16.0,
            "EURUSD=X": 1.1,
            "^TNX": 4.2,
            "NQ=F": 20000.0,
            "ES=F": 6000.0,
        }
        with patch.object(
            market_data,
            "download_close",
            side_effect=lambda ticker, _: pd.Series([latest_values[ticker]]),
        ) as download:
            result = market_data.macro_context_agent(
                {},
                indicators=configured_indicators,
            )

        context = result["macro_context"]
        self.assertEqual(
            context["enabled_indicators"],
            ["vix", "eur_usd", "us10y", "nasdaq_futures", "sp500_futures"],
        )
        self.assertEqual(set(context["indicators"]), set(context["enabled_indicators"]))
        self.assertEqual(context["risk_mode"], "RISK_ON")
        self.assertEqual(
            [call.args[0] for call in download.call_args_list],
            ["^VIX", "EURUSD=X", "^TNX", "NQ=F", "ES=F"],
        )

    def test_configured_ollama_adds_separate_summary(self) -> None:
        response = unittest.mock.Mock()
        response.json.return_value = {
            "message": {"content": "Zusammenfassung auf Basis der Kennzahlen."}
        }
        config = test_config(ollama_active=True)
        with (
            patch.object(market_data, "download_ohlc", side_effect=self._download),
            patch.object(agents.requests, "post", return_value=response) as post,
        ):
            result = graph.build_graph("^GDAXI", config).invoke({})

        self.assertIn("MARKET DAILY REVIEW", result["report"])
        self.assertEqual(
            result["llm_summary"],
            "Zusammenfassung auf Basis der Kennzahlen.",
        )
        post.assert_called_once()
        self.assertEqual(
            post.call_args.args[0],
            f"{config.ollama.base_url}/api/chat",
        )
        self.assertEqual(
            post.call_args.kwargs["json"]["model"],
            config.ollama.model,
        )

    def test_disabled_ollama_skips_api_even_when_url_is_configured(self) -> None:
        config = test_config(ollama_active=False)
        with (
            patch.object(market_data, "download_ohlc", side_effect=self._download),
            patch.object(agents.requests, "post") as post,
        ):
            result = graph.build_graph("^GDAXI", config).invoke({})

        self.assertIn("MARKET DAILY REVIEW", result["report"])
        self.assertNotIn("llm_summary", result)
        post.assert_not_called()

    def test_news_is_reported_without_changing_quant_or_risk(self) -> None:
        baseline_config = test_config()
        settings = config_module.NewsConfig(
            active=True,
            lookback_hours=24,
            max_articles=10,
            gdelt_active=True,
            gdelt_query="Trump market",
            rss_feeds=(),
        )
        config = config_module.AppConfig(
            ollama=baseline_config.ollama,
            macro_indicators=baseline_config.macro_indicators,
            news=settings,
        )
        response = unittest.mock.Mock()
        response.json.return_value = {
            "articles": [
                {
                    "title": "Trump tariff deal boosts stocks as markets rally",
                    "url": "https://example.com/trump-market-story",
                    "seendate": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
                    "domain": "example.com",
                    "language": "English",
                }
            ]
        }

        with patch.object(market_data, "download_ohlc", side_effect=self._download):
            baseline = graph.build_graph("^ATX", baseline_config).invoke({})
            with patch.object(
                news,
                "_request_gdelt",
                return_value=response.json.return_value,
            ):
                result = graph.build_graph("^ATX", config).invoke({})

        self.assertEqual(result["quant"], baseline["quant"])
        self.assertEqual(result["risk"], baseline["risk"])
        self.assertIn("Trump-related news context", result["report"])
        self.assertIn("Headline sentiment (keyword heuristic; informational only): POSITIVE", result["report"])
        self.assertIn("Provider: GDELT | Source: example.com", result["report"])
        self.assertIn("not used in quant or risk signal", result["report"])


class NewsAgentTests(unittest.TestCase):
    def test_gdelt_uses_curl_with_encoded_parameters_and_bounded_retries(self) -> None:
        payload = {"articles": []}
        result = unittest.mock.Mock(
            returncode=0,
            stdout=f"{json.dumps(payload)}{news.GDELT_STATUS_MARKER}200",
            stderr="",
        )
        params = {
            "query": "Trump (market OR stocks)",
            "mode": "ArtList",
            "format": "json",
            "maxrecords": 10,
            "sort": "DateDesc",
            "timespan": "24h",
        }

        with patch.object(news.subprocess, "run", return_value=result) as run:
            response = news._request_gdelt(params)

        self.assertEqual(response, payload)
        command = run.call_args.args[0]
        self.assertEqual(command[0], "curl")
        self.assertIn("query=Trump (market OR stocks)", command)
        self.assertIn("--connect-timeout", command)
        self.assertEqual(command[command.index("--connect-timeout") + 1], "10")
        self.assertEqual(command[command.index("--max-time") + 1], "130")
        self.assertEqual(command[command.index("--retry") + 1], "2")
        self.assertEqual(command[command.index("--retry-delay") + 1], "10")
        self.assertEqual(run.call_args.kwargs["timeout"], 400)

    def test_gdelt_curl_http_errors_are_reported(self) -> None:
        result = unittest.mock.Mock(
            returncode=0,
            stdout=f"rate limited{news.GDELT_STATUS_MARKER}429",
            stderr="",
        )

        with patch.object(news.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(news.requests.HTTPError, "HTTP 429: rate limited"):
                news._request_gdelt({"query": "Trump"})

    def test_gdelt_curl_timeouts_are_reported(self) -> None:
        with patch.object(
            news.subprocess,
            "run",
            side_effect=news.subprocess.TimeoutExpired("curl", 400),
        ):
            with self.assertRaisesRegex(news.requests.Timeout, "exceeded 400 seconds"):
                news._request_gdelt({"query": "Trump"})

    def test_news_agent_combines_recent_gdelt_and_rss_headlines(self) -> None:
        now = datetime.now(UTC).replace(microsecond=0)
        recent = now.strftime("%Y%m%dT%H%M%SZ")
        old = (now - timedelta(days=2)).strftime("%Y%m%dT%H%M%SZ")
        gdelt_response = unittest.mock.Mock()
        gdelt_response.json.return_value = {
            "articles": [
                {
                    "title": "Trump tariff deal boosts stocks as markets rally",
                    "url": "https://example.com/trump-market-story?utm_source=feed",
                    "seendate": recent,
                    "domain": "example.com",
                    "language": "English",
                },
                {
                    "title": "Trump tariffs caused markets to rally last week",
                    "url": "https://example.com/old",
                    "seendate": old,
                    "domain": "example.com",
                    "language": "English",
                },
                {
                    "title": "Trump gives a speech",
                    "url": "https://example.com/unrelated",
                    "seendate": recent,
                    "domain": "example.com",
                    "language": "English",
                },
                {
                    "title": "Trump tariff threatens stocks",
                    "url": "https://example.com/foreign",
                    "seendate": recent,
                    "domain": "example.com",
                    "language": "Chinese",
                },
            ]
        }
        rss_response = unittest.mock.Mock()
        rss_response.content = f"""<?xml version="1.0"?>
            <rss version="2.0"><channel><item>
              <title>Trump threatens new tariffs as stocks plunge</title>
              <description>Trade policy and markets react</description>
              <link>https://rss.example.com/trump-tariffs</link>
              <pubDate>{now.strftime("%a, %d %b %Y %H:%M:%S GMT")}</pubDate>
            </item></channel></rss>""".encode()
        settings = config_module.NewsConfig(
            active=True,
            lookback_hours=24,
            max_articles=10,
            gdelt_active=True,
            gdelt_query="Trump market",
            rss_feeds=(
                config_module.NewsFeedConfig(
                    name="Test RSS",
                    url="https://rss.example.com/feed",
                    active=True,
                ),
            ),
        )

        with (
            patch.object(
                news,
                "_request_gdelt",
                return_value=gdelt_response.json.return_value,
            ) as gdelt_request,
            patch.object(news.requests, "get", return_value=rss_response) as get,
        ):
            context = news.news_context_agent({}, settings=settings)["news_context"]

        self.assertEqual(len(context["articles"]), 2)
        self.assertEqual(
            {article["provider"] for article in context["articles"]},
            {"GDELT", "Test RSS"},
        )
        gdelt_article = next(
            article for article in context["articles"] if article["provider"] == "GDELT"
        )
        self.assertEqual(gdelt_article["source"], "example.com")
        self.assertEqual(context["overall_sentiment"], "MIXED")
        self.assertEqual(
            {article["sentiment"] for article in context["articles"]},
            {"POSITIVE", "NEGATIVE"},
        )
        self.assertTrue(
            any("trade_policy" in article["topics"] for article in context["articles"])
        )
        self.assertEqual(
            {
                article["sentiment_source"]
                for article in context["articles"]
            },
            {"keyword heuristic"},
        )
        gdelt_request.assert_called_once()
        self.assertEqual(gdelt_request.call_args.args[0]["timespan"], "24h")
        self.assertEqual(get.call_count, 1)

    def test_news_errors_are_retained_in_context(self) -> None:
        settings = config_module.NewsConfig(
            active=True,
            lookback_hours=24,
            max_articles=10,
            gdelt_active=True,
            gdelt_query="Trump market",
            rss_feeds=(),
        )
        with patch.object(
            news,
            "_request_gdelt",
            side_effect=agents.requests.HTTPError("429 rate limit"),
        ):
            context = news.news_context_agent({}, settings=settings)["news_context"]

        self.assertEqual(context["errors"], {"GDELT": "429 rate limit"})
        self.assertTrue(context["active"])

    def test_ollama_classifies_market_impact_as_structured_json(self) -> None:
        now = datetime.now(UTC)
        article = news._make_article(
            provider="Test RSS",
            source="Test",
            title="Trump considers military action after deadly airport strike",
            description="Oil markets react to geopolitical risks.",
            url="https://example.com/news",
            published_at=now.strftime("%a, %d %b %Y %H:%M:%S GMT"),
            cutoff=now - timedelta(hours=24),
            now=now,
        )
        self.assertIsNotNone(article)
        ollama_response = unittest.mock.Mock()
        ollama_response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {
                        "articles": [
                            {
                                "index": 0,
                                "sentiment": "NEGATIVE",
                                "market_impact": "RISK_OFF",
                                "impact_reason": (
                                    "Potential military escalation may raise "
                                    "geopolitical risk and energy uncertainty."
                                ),
                                "topics": ["geopolitics", "markets_economy"],
                            }
                        ]
                    }
                )
            }
        }
        config = config_module.NewsConfig(
            active=True,
            lookback_hours=24,
            max_articles=10,
            gdelt_active=False,
            gdelt_query="Trump market",
            rss_feeds=(
                config_module.NewsFeedConfig("Test RSS", "https://example.com/rss", True),
            ),
        )
        rss_response = unittest.mock.Mock()
        rss_response.content = f"""<rss><channel><item>
            <title>Trump considers military action after deadly airport strike</title>
            <description>Oil markets react to geopolitical risks.</description>
            <link>https://example.com/news</link>
            <pubDate>{now.strftime("%a, %d %b %Y %H:%M:%S GMT")}</pubDate>
            </item></channel></rss>""".encode()
        ollama_config = config_module.OllamaConfig(
            active=True,
            base_url="http://ollama.local:11434",
            model="qwen3:14b",
        )

        with patch.object(
            news.requests,
            "get",
            return_value=rss_response,
        ), patch.object(
            news.requests,
            "post",
            return_value=ollama_response,
        ) as post:
            context = news.news_context_agent(
                {},
                settings=config,
                ollama_settings=ollama_config,
            )["news_context"]

        self.assertEqual(context["sentiment_method"], "Ollama (qwen3:14b)")
        self.assertEqual(context["overall_sentiment"], "NEGATIVE")
        classified_article = context["articles"][0]
        self.assertEqual(classified_article["sentiment"], "NEGATIVE")
        self.assertEqual(classified_article["market_impact"], "RISK_OFF")
        self.assertEqual(classified_article["sentiment_source"], "Ollama (qwen3:14b)")
        self.assertIn("military escalation", classified_article["impact_reason"])
        self.assertEqual(post.call_args.kwargs["json"]["format"], "json")
        self.assertEqual(post.call_args.kwargs["timeout"], (5, 60))

    def test_invalid_ollama_result_is_reported_and_uses_heuristics(self) -> None:
        now = datetime.now(UTC)
        settings = config_module.NewsConfig(
            active=True,
            lookback_hours=24,
            max_articles=10,
            gdelt_active=False,
            gdelt_query="Trump market",
            rss_feeds=(
                config_module.NewsFeedConfig("Test RSS", "https://example.com/rss", True),
            ),
        )
        rss_response = unittest.mock.Mock()
        rss_response.content = f"""<rss><channel><item>
            <title>Trump considers military action after deadly airport strike</title>
            <description>Oil markets react to geopolitical risks.</description>
            <link>https://example.com/news</link>
            <pubDate>{now.strftime("%a, %d %b %Y %H:%M:%S GMT")}</pubDate>
            </item></channel></rss>""".encode()
        ollama_response = unittest.mock.Mock()
        ollama_response.json.return_value = {"message": {"content": "not JSON"}}
        ollama_config = config_module.OllamaConfig(
            active=True,
            base_url="http://ollama.local:11434",
            model="qwen3:14b",
        )

        with patch.object(
            news.requests,
            "get",
            return_value=rss_response,
        ), patch.object(
            news.requests,
            "post",
            return_value=ollama_response,
        ):
            context = news.news_context_agent(
                {},
                settings=settings,
                ollama_settings=ollama_config,
            )["news_context"]

        self.assertEqual(context["sentiment_method"], "keyword heuristic")
        self.assertIn("Ollama sentiment", context["errors"])
        self.assertIn("using keyword heuristic", context["errors"]["Ollama sentiment"])
        self.assertEqual(context["articles"][0]["sentiment"], "NEGATIVE")
        self.assertEqual(context["articles"][0]["market_impact"], "RISK_OFF")


class ConfigTests(unittest.TestCase):
    def test_project_config_uses_configured_ollama_endpoint(self) -> None:
        config = config_module.load_config()

        self.assertTrue(config.ollama.base_url.startswith("http"))
        self.assertEqual(config.ollama.model, "qwen3:14b")
        self.assertEqual(
            {indicator.name for indicator in config.macro_indicators},
            {"vix", "eur_usd", "us10y", "nasdaq_futures", "sp500_futures"},
        )
        self.assertEqual(
            {indicator.ticker for indicator in config.macro_indicators},
            {"^VIX", "EURUSD=X", "^TNX", "NQ=F", "ES=F"},
        )
        self.assertIsNotNone(config.news)
        assert config.news is not None
        self.assertTrue(config.news.active)
        self.assertTrue(config.news.gdelt_active)
        self.assertEqual(
            {feed.name for feed in config.news.rss_feeds},
            {"CNBC US Top News", "MarketWatch Top Stories"},
        )

    def test_config_rejects_non_http_ollama_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ollama": {
                            "active": False,
                            "base_url": "ftp://ollama.local",
                            "model": "llama3.3",
                        },
                        "macro_indicators": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "HTTP\\(S\\) URL"):
                config_module.load_config(config_path)

    def test_ollama_can_be_disabled_even_with_url_configured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ollama": {
                            "active": False,
                            "base_url": "http://ollama.local:11434",
                            "model": "qwen3:14b",
                        },
                        "macro_indicators": {},
                    }
                ),
                encoding="utf-8",
            )
            config = config_module.load_config(config_path)

        self.assertFalse(config.ollama.active)
        self.assertEqual(config.ollama.base_url, "http://ollama.local:11434")


if __name__ == "__main__":
    unittest.main()
