import requests

from trading_agent.config import OllamaConfig
from trading_agent.state import RiskData, TradingState


def risk_agent(state: TradingState) -> dict[str, RiskData]:
    quant = state.get("quant")
    macro = state.get("macro_context")
    if quant is None or macro is None:
        raise RuntimeError("Quant or macro context is missing from graph state.")

    if quant["bias"] == "NEUTRAL":
        candidate = False
        reason = "No sufficiently strong directional signal; manual review not suggested."
    elif "vix" in macro["enabled_indicators"] and "vix" not in macro["indicators"]:
        candidate = False
        reason = "VIX is configured but unavailable; the volatility risk filter cannot assess the signal."
    elif "vix" in macro["indicators"] and macro["indicators"]["vix"] >= 30:
        candidate = False
        reason = "VIX is at or above 30; the high-volatility filter blocks this signal."
    elif "vix" in macro["indicators"] and macro["indicators"]["vix"] >= 20:
        candidate = True
        reason = "Signal passed basic filters, but VIX is elevated; manual review advised."
    elif "vix" not in macro["enabled_indicators"]:
        candidate = True
        reason = "VIX filter is disabled; manual review required without that volatility check."
    else:
        candidate = True
        reason = "Signal passed basic MVP filters; manual review required."

    return {"risk": {"review_candidate": candidate, "reason": reason}}


def report_agent(state: TradingState) -> dict[str, str]:
    market = state.get("market_data")
    macro = state.get("macro_context")
    quant = state.get("quant")
    risk = state.get("risk")
    news = state.get("news_context")
    if any(item is None for item in (market, macro, quant, risk)):
        raise RuntimeError("Cannot create report; required analysis data is missing.")

    macro_lines = []
    for name in macro["enabled_indicators"]:
        if name in macro["indicators"]:
            macro_lines.append(f"{name}: {macro['indicators'][name]:.4f}")
        else:
            macro_lines.append(f"{name}: UNAVAILABLE ({macro['errors'].get(name, 'no data')})")
    if not macro_lines:
        macro_lines.append("No macro indicators enabled.")

    if news is None:
        news_lines = ["News agent did not return a result."]
    elif not news["active"]:
        news_lines = ["News agent is disabled."]
    else:
        news_lines = [
            f"Headline sentiment ({news['sentiment_method']}; informational only): "
            f"{news['overall_sentiment']}"
        ]
        if news["enabled_sources"] and not news["articles"]:
            news_lines.append(f"Sources: {', '.join(news['enabled_sources'])}")
        if news["articles"]:
            for index, article in enumerate(news["articles"]):
                source_label = (
                    f" | Source: {article['source']}"
                    if article["source"] != article["provider"]
                    else ""
                )
                topics = ", ".join(article["topics"]) or "market policy"
                if index:
                    news_lines.append("")
                news_lines.extend(
                    [
                        f"- [{article['published_at']}] Provider: "
                        f"{article['provider']}{source_label}",
                        f"  News: {article['title']}",
                        f"  Sentiment: {article['sentiment']}",
                        f"  Potential market impact: {article['market_impact']}",
                        f"  Topics: {topics}",
                        f"  Assessment: {article['impact_reason']}",
                        f"  URL: {article['url']}",
                    ]
                )
        else:
            news_lines.append("No recent Trump-related market-impact headlines found.")
        news_lines.append(
            "Potential market impact is an assessment, not a price forecast; "
            "news does not change the quant or risk signal."
        )
    if news is not None:
        for source, error in news["errors"].items():
            news_lines.append(f"Source error ({source}): {error}")

    report = f"""MARKET DAILY REVIEW
    
Symbol: {market['symbol']}
Latest available daily candle: {market['date']}

Daily open: {market['open']:.2f}
Close: {market['close']:.2f}
EMA20: {market['ema20']:.2f}
EMA50: {market['ema50']:.2f}
EMA200: {market['ema200']:.2f}
RSI14: {market['rsi']:.2f}
ATR14: {market['atr14']:.2f}
Gap from previous close: {market['gap_pct']:.2f}%

Long score: {quant['long_score']}
Short score: {quant['short_score']}
Quant bias: {quant['bias']}

Macro / market context ({macro['risk_mode']}):
{chr(10).join(macro_lines)}

Trump-related news context (not used in quant or risk signal):
{chr(10).join(news_lines)}

Manual-review candidate: {risk['review_candidate']}
Risk note: {risk['reason']}

This is an experimental, rule-based market review. It is not a validated
strategy, an order, or investment advice.
"""
    return {"report": report}


def ollama_summary_agent(
    state: TradingState,
    *,
    settings: OllamaConfig,
) -> dict[str, str]:
    report = state.get("report")
    if not report:
        raise RuntimeError("Deterministic report is missing; cannot summarize with Ollama.")

    response = requests.post(
        f"{settings.base_url}/api/chat",
        json={
            "model": settings.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Fasse den folgenden Marktbericht knapp auf Deutsch zusammen. "
                        "Verwende ausschließlich die enthaltenen Fakten. Ändere weder "
                        "Bias noch Risikobewertung und gib keine Orderempfehlung."
                    ),
                },
                {"role": "user", "content": report},
            ],
        },
        timeout=(5, 120),
    )
    response.raise_for_status()
    payload = response.json()
    summary = payload.get("message", {}).get("content")
    if not isinstance(summary, str) or not summary.strip():
        raise RuntimeError("Ollama returned an empty or invalid summary.")
    return {"llm_summary": summary.strip()}
