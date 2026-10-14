from trading_agent.config import load_config
from trading_agent.graph import build_graph


def main(symbol: str) -> None:
    config = load_config()
    result = build_graph(symbol=symbol, config=config).invoke({})
    print(result["report"])
    if summary := result.get("llm_summary"):
        print("OLLAMA SUMMARY\n")
        print(summary)


if __name__ == "__main__":
    main("^GDAXI")
