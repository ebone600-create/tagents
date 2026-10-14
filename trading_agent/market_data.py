import math

import pandas as pd
import ta
import yfinance as yf

from trading_agent.config import MacroIndicatorConfig
from trading_agent.state import MacroContext, MarketData, TradingState


def download_ohlc(ticker: str, period: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {ticker}.")

    fields = ["Open", "High", "Low", "Close"]
    if isinstance(data.columns, pd.MultiIndex):
        selected_columns = []
        for field in fields:
            matches = [column for column in data.columns if field in column]
            ticker_match = next(
                (column for column in matches if ticker in column),
                None,
            )
            if ticker_match is not None:
                selected_columns.append(ticker_match)
            elif len(matches) == 1:
                selected_columns.append(matches[0])
            else:
                raise ValueError(
                    f"Could not identify a unique {field} column for {ticker}."
                )
        ohlc = data.loc[:, selected_columns].copy()
        ohlc.columns = fields
    else:
        missing = set(fields) - set(data.columns)
        if missing:
            raise ValueError(
                f"Missing OHLC columns for {ticker}: {', '.join(sorted(missing))}."
            )
        ohlc = data.loc[:, fields].copy()

    for field in fields:
        ohlc[field] = pd.to_numeric(ohlc[field], errors="coerce")
    ohlc = ohlc.dropna()
    if ohlc.empty:
        raise RuntimeError(f"Yahoo Finance returned no valid OHLC data for {ticker}.")
    return ohlc


def download_close(ticker: str, period: str) -> pd.Series:
    return download_ohlc(ticker, period)["Close"]


def load_market_data(
    _: TradingState,
    *,
    symbol: str,
) -> dict[str, MarketData]:
    if not symbol.strip():
        raise ValueError("The market symbol must not be empty.")
    prices = download_ohlc(symbol, "2y")
    if len(prices) < 200:
        raise RuntimeError(
            f"Need at least 200 observations for {symbol}; got {len(prices)}."
        )

    open_price = prices["Open"]
    high = prices["High"]
    low = prices["Low"]
    close = prices["Close"]
    metrics = pd.DataFrame(
        {
            "open": open_price,
            "close": close,
            "ema20": ta.trend.ema_indicator(close, window=20),
            "ema50": ta.trend.ema_indicator(close, window=50),
            "ema200": ta.trend.ema_indicator(close, window=200),
            "rsi": ta.momentum.rsi(close, window=14),
            "atr14": ta.volatility.average_true_range(
                high,
                low,
                close,
                window=14,
            ),
            "gap_pct": (open_price / close.shift(1) - 1) * 100,
        }
    ).dropna()
    if metrics.empty:
        raise RuntimeError(f"Not enough valid {symbol} data to calculate indicators.")

    latest = metrics.iloc[-1]
    values = {name: float(latest[name]) for name in metrics.columns}
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError(f"Latest {symbol} indicators contain non-finite values.")

    return {
        "market_data": {
            "symbol": symbol,
            "date": pd.Timestamp(metrics.index[-1]).date().isoformat(),
            **values,
        }
    }


def macro_context_agent(
    _: TradingState,
    *,
    indicators: tuple[MacroIndicatorConfig, ...],
) -> dict[str, MacroContext]:
    enabled_indicators = [item for item in indicators if item.active]
    values: dict[str, float] = {}
    errors: dict[str, str] = {}

    for indicator in enabled_indicators:
        try:
            close = download_close(indicator.ticker, "5d")
            value = float(close.iloc[-1])
            if not math.isfinite(value):
                raise RuntimeError("latest value is not finite")
            values[indicator.name] = value
        except (RuntimeError, ValueError, TypeError, OSError) as error:
            errors[indicator.name] = str(error)

    if "vix" in errors:
        risk_mode = "UNAVAILABLE"
    elif "vix" not in values:
        risk_mode = "UNAVAILABLE"
    elif values["vix"] < 20:
        risk_mode = "RISK_ON"
    elif values["vix"] < 30:
        risk_mode = "CAUTIOUS"
    else:
        risk_mode = "RISK_OFF"

    return {
        "macro_context": {
            "indicators": values,
            "errors": errors,
            "enabled_indicators": [item.name for item in enabled_indicators],
            "risk_mode": risk_mode,
        }
    }
