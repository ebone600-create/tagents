from functools import partial
from typing import Literal

from langgraph.graph import END, START, StateGraph

from trading_agent.agents import ollama_summary_agent, report_agent, risk_agent
from trading_agent.analysis import quant_agent
from trading_agent.config import AppConfig, OllamaConfig
from trading_agent.market_data import load_market_data, macro_context_agent
from trading_agent.news import news_context_agent
from trading_agent.state import TradingState


def _ollama_is_configured(
    _: TradingState,
    *,
    settings: OllamaConfig,
) -> Literal["ollama_summary", "done"]:
    return "ollama_summary" if settings.active else "done"


def build_graph(symbol: str, config: AppConfig):
    if not symbol.strip():
        raise ValueError("The market symbol must not be empty.")

    builder = StateGraph(TradingState)
    builder.add_node("market_data", partial(load_market_data, symbol=symbol))
    builder.add_node(
        "macro_context",
        partial(macro_context_agent, indicators=config.macro_indicators),
    )
    builder.add_node(
        "news_context",
        partial(
            news_context_agent,
            settings=config.news,
            ollama_settings=config.ollama,
        ),
    )
    builder.add_node("quant", quant_agent)
    builder.add_node("risk", risk_agent)
    builder.add_node("report", report_agent)
    if config.ollama.active:
        builder.add_node(
            "ollama_summary",
            partial(ollama_summary_agent, settings=config.ollama),
        )

    builder.add_edge(START, "market_data")
    builder.add_edge(START, "macro_context")
    builder.add_edge(START, "news_context")
    builder.add_edge(["market_data", "macro_context"], "quant")
    builder.add_edge("quant", "risk")
    builder.add_edge(["risk", "news_context"], "report")
    if config.ollama.active:
        builder.add_conditional_edges(
            "report",
            partial(_ollama_is_configured, settings=config.ollama),
            {"ollama_summary": "ollama_summary", "done": END},
        )
        builder.add_edge("ollama_summary", END)
    else:
        builder.add_edge("report", END)
    return builder.compile()
