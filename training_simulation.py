# agentic_research_ai.py

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableLambda
from typing import Dict,List

from llm import evaluate_customer_response , get_customer_reply

# === Agent Function ===
def simulate_customer_interaction(state: dict) -> dict:
    history: List[Dict[str, str]] = []

    # Initial customer message
    initial_customer = "Hello, I need to return my order. Can you help?"
    print(f"🧑 Customer: {initial_customer}")
    history.append({"role": "user", "content": initial_customer})

    while True:
        user_input = input("💬 Your response: ")
        history.append({"role": "assistant", "content": user_input})

        feedback = evaluate_customer_response(history)
        print("🤖 Feedback:", feedback["feedback"])

        if feedback.get("complete"):
            break

        if feedback.get("passed"):
            customer_reply = get_customer_reply(history)
            print(f"🧑 Customer: {customer_reply}")
            history.append({"role": "user", "content": customer_reply})
        else:
            step = feedback.get("step")
            if step:
                print("⚠️ Please revise your response to meet the current SOP step. ("+step+")")
            else:
                print("⚠️ Please revise your response to meet the current SOP step. (Unknown step)")

        # Generate next customer reply based on updated history
        customer_reply = get_customer_reply(history)
        print(f"🧑 Customer: {customer_reply}")
        history.append({"role": "user", "content": customer_reply})

    return state

# === Workflow Definition ===
def create_graph():
    builder = StateGraph(dict)

    builder.add_node("simulate_customer_interaction", RunnableLambda(simulate_customer_interaction))
    builder.set_entry_point("simulate_customer_interaction")
    builder.add_edge("simulate_customer_interaction", END)

    return builder.compile()

# === Entrypoint ===
if __name__ == "__main__":
    user_query = "Handle missing order inquiry."
    initial_state = {"query": user_query}

    graph = create_graph()
    final_state: dict = graph.invoke(initial_state)

    print("\n✅ Customer simulation passed.")
