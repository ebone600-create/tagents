# Generic market review agent MVP

This first milestone produces a daily, rule-based DAX review. It does not place
orders or calculate an executable position size.

## Included

- Downloads daily OHLC data for the selected market symbol and configured
  macro-context symbols from Yahoo Finance.
- The root `dax_agents.py` entry point currently selects the DAX (`^GDAXI`).
  Change `SYMBOL` there to another Yahoo Finance symbol, for example `^ATX`.
- Configurable macro context defaults include VIX, EUR/USD, US 10-year Treasury
  yield, Nasdaq futures, and S&P 500 futures. Enable or disable entries in
  `config.json`; VIX alone currently affects the simple risk filter.
- A separate news branch reads recent market-related Trump headlines from
  GDELT and configured RSS feeds (CNBC and MarketWatch by default). Headlines
  are filtered by date and topic, deduplicated, and classified by Ollama when
  `ollama.active` is true. If Ollama is disabled or its classification response
  is invalid, a transparent keyword heuristic is used and the problem is
  reported. Each story shows both its tone and a cautious potential market-impact
  assessment. News does not change the quant bias or risk decision.
- Fetches the independent data branches in parallel through LangGraph.
- Calculates EMA20, EMA50, EMA200, RSI14, ATR14, and the daily opening gap.
- Loads two years of daily OHLC data to provide sufficient history for EMA200.
  Gap % is today's open versus the previous daily close.
- Derives transparent long/short scores from EMA20, EMA50, and RSI14. The
  additional indicators are reported but do not change the signal yet.
- Blocks manual-review candidates when VIX is at least 30; elevated VIX adds a
  warning.
- Produces a deterministic report. If configured, Ollama adds a separate text
  summary; it cannot alter the quant signal or risk result.

## Run

From the project root, run:

```bash
source tagents-env/bin/activate
python -m trading_agent.main
```

The previous root-level command remains available as a compatibility entry
point:

```bash
python dax_agents.py
```

Run the tests from the project root with:

```bash
python -m unittest discover -v
```

## Project layout

- `trading_agent/config.py`: loads and validates local configuration.
- `trading_agent/market_data.py`: retrieves OHLC data and computes market metrics.
- `trading_agent/news.py`: retrieves, filters, and labels GDELT/RSS headlines.
- `trading_agent/analysis.py`: derives the rule-based quant signal.
- `trading_agent/agents.py`: risk filter, report, and optional Ollama summary.
- `trading_agent/graph.py`: connects the LangGraph nodes and edges.
- `trading_agent/main.py`: application entry point.
- `trading_agent/state.py`: shared state and data types.
- `dax_agents.py`: selects the primary symbol and starts the generic agent.

Yahoo Finance and the news feeds are used for this prototype and are not
guaranteed production data sources. GDELT limits request frequency; the agent
makes one GDELT request per run and reports provider errors instead of hiding
them. Check each publisher's current feed terms before using its content beyond
private research. The keyword fallback is not a validated sentiment model. Ollama's market-impact
assessment is also contextual analysis, not a price forecast or investment
advice. The news branch uses the configured Ollama model only when
`ollama.active` is true; the same switch controls the separate report summary.

## Optional Ollama summary

Edit `config.json` to control Ollama. Set `ollama.active` to `false` to skip the
LLM while developing other functionality, even if the URL remains configured.
When active, configure
`ollama.base_url` with the Ollama API base URL and `ollama.model` with a model
installed on that host. The current configuration uses `qwen3:14b`. Ollama
errors are surfaced; they do not silently fall back to a success-shaped report.

Set the primary instrument in `SYMBOL` in `dax_agents.py`. For example, use
`^GDAXI` for the DAX or `^ATX` for the ATX. Then run from the project root:

```bash
source tagents-env/bin/activate
python dax_agents.py
```

## Deliberately deferred

Intraday VWAP and Opening Range need intraday OHLCV data and an explicit session
definition; they are deferred to a TradingView-data milestone. DuckDB/Polars persistence, VectorBT
backtesting, XGBoost, and a scheduler belong in later milestones after the input
data and strategy assumptions are defined. They are not dependencies of this
MVP.
