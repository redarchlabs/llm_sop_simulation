# agentic_research_ai.py

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableLambda
from typing import List, Dict, Any

from tools import ResearchTools  # <- you will implement this module
from llm import summarize_chunks  # <- simple summarizer using Ollama LLM


# === Agent Functions ===
def retrieve_chunks(state: dict, tools: ResearchTools) -> dict:
    results = tools.search_vector_db(state["query"], top_k=10)
    state["retrieved_chunks"] = results
    return state

def summarize(state: dict) -> dict:
    state["summary"] = summarize_chunks(state["retrieved_chunks"])
    return state

# === Workflow Definition ===
def create_graph(tools):
    builder = StateGraph(dict)

    builder.add_node("retrieve_chunks", RunnableLambda(lambda s: retrieve_chunks(s, tools)))
    builder.add_node("summarize", RunnableLambda(summarize))

    builder.set_entry_point("retrieve_chunks")
    builder.add_edge("retrieve_chunks", "summarize")
    builder.add_edge("summarize", END)

    return builder.compile()

# === Entrypoint ===
if __name__ == "__main__":
    from drivers import qdrant_driver, graph_driver  
    tools = ResearchTools(qdrant_driver, graph_driver)

    user_query = "Can you provide some really good examples of email responses from JBAF_LAW to customers?  Do not invent or infer additional examples.  Just provide the best examples you can find in the context. Only select emails that are actual responses to customer messages — these should clearly reference a prior message, contain 'Re:' in the subject, or include phrases like 'following up', 'as requested', or 'regarding your inquiry'."

    graph = create_graph(tools)
    initial_state = {"query": user_query}
    final_state = graph.invoke(initial_state)

    print("\n\n📬 Final Summary:\n")
    print(final_state["summary"])

