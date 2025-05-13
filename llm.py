# llm.py

import ollama
import json
import requests

# fast_vectorizer.py
from sentence_transformers import SentenceTransformer

# Choose a fast, local embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(texts:str)-> list:
    """
    Get the embedding for a given text or list of texts."""
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(texts).tolist()

# def get_embedding(text: str) -> list:
#     # Assumes Ollama model supports embedding, like 'nomic-embed-text'
#     response = ollama.embeddings(model="nomic-embed-text", prompt=text)
#     return response["embedding"]

def summarize_chunks(chunks: list) -> str:
    context = "\n\n".join(chunk["text"] for chunk in chunks)
    prompt = f"""
You are an assistant reviewing customer service emails. The following are snippets of good responses to customers.

Please extract the best 3 examples and explain why they are good.

--- BEGIN TEXT ---
{context}
--- END TEXT ---
"""
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def chat_with_llm(prompt: str) -> str:
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def is_reply_chunk(text: str) -> bool:
    prompt = f"""
Is the following email chunk a REPLY to a customer?

Look for signs like 'Re:' in the subject line, reply-style greetings, or references to prior conversations.

--- EMAIL CHUNK ---
{text}
--------------------

Reply with ONLY 'yes' or 'no'.
"""
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip().lower() == "yes"


def format_history_for_llm(history: list) -> str:
    return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])

def get_customer_reply(history: list) -> str:
    prompt = f"""
You are the customer in this conversation. Respond realistically based on the latest message from the customer service agent.

Here is the full conversation history:

{format_history_for_llm(history)}

Respond with your next message as the customer.
"""

    return chat_with_llm(prompt)


def evaluate_customer_response(history: list) -> dict:
    prompt = f"""
You are a customer service trainer evaluating the trainee's performance based on SOP145.

SOP145 requires:
- Acknowledge the issue with empathy
- Ask for order details and relevant evidence
- Check refund eligibility
- Set refund timeline expectations
- Provide status updates
- Confirm refund completion

Below is the full conversation so far:

{format_history_for_llm(history)}

Evaluate whether the trainee has addressed ALL required items in SOP145 so far. If not, explain what is missing.

Respond in VALID JSON with:
- "step": description of the current SOP step being addressed
- "passed": true if the current step is complete, false otherwise
- "complete": true if all steps are complete
- "feedback": a short explanation of what was good or what’s missing in bulleted list format (easy to digest).  Recommend specific improvements.
- "examples": a list of examples from the conversation that demonstrate good or bad practices.
Be specific and provide examples from the conversation to support your evaluation.
Always provide the expected tasks as a numbered list.
- "expected_tasks": a numbered list of tasks that should have been completed according to SOP145.
- "actual_tasks": a numbered list of tasks that were actually completed by the trainee.

It is critical your response is in JSON format. Do not include any other text or explanations outside of the JSON.
"""

    content = chat_with_llm(prompt)
    if not content:
        return {"passed": False, "feedback": "No response from LLM."}
    try:
        return json.loads(content)
    except:
        return {"passed": False, "feedback": "Could not parse response: " + content}
