from typing import Literal

from trading_agent.state import QuantData, TradingState


def quant_agent(state: TradingState) -> dict[str, QuantData]:
    data = state.get("market_data")
    if data is None:
        raise RuntimeError("Market data is missing from graph state.")

    long_score = 0
    short_score = 0
    if data["close"] > data["ema20"]:
        long_score += 30
    elif data["close"] < data["ema20"]:
        short_score += 30

    if data["ema20"] > data["ema50"]:
        long_score += 40
    elif data["ema20"] < data["ema50"]:
        short_score += 40

    if 50 <= data["rsi"] < 70:
        long_score += 30
    elif 30 < data["rsi"] <= 50:
        short_score += 30

    if long_score >= 70 and long_score > short_score:
        bias: Literal["LONG", "SHORT", "NEUTRAL"] = "LONG"
    elif short_score >= 70 and short_score > long_score:
        bias = "SHORT"
    else:
        bias = "NEUTRAL"

    return {
        "quant": {
            "long_score": long_score,
            "short_score": short_score,
            "bias": bias,
        }
    }
