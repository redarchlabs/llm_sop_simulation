Below is a checklist of must‑have prompt elements for keeping a “customer‑return” simulation on‑track, plus a concise skeleton you can adapt. If you include each item, you’ll drastically reduce drift, hallucination, or tone problems.

#	Prompt Element	Why it matters	Tips / Sample wording
1	Roles & persona definitions	Prevents the model from switching viewpoints.	“You are Jamie, a frustrated customer returning a defective pair of headphones.”
2	Objective statement	Gives the model a clear success criterion.	“Your goal is to obtain a full refund or exchange, following the store’s return policy.”
3	Conversation scope	Stops topic drift.	“Stay strictly on the topic of returning this item; do not discuss unrelated products or policies.”
4	Required interaction steps / SOP checkpoints	Ensures the trainee is tested on every critical skill and lets you halt the scenario if the step isn’t passed.	List the steps in order (e.g. 1 Empathy → 2 Gather order details → 3 Confirm policy → 4 Outline next actions). Include “Do not advance to the next step until the employee response meets the criteria.”
5	Tone & style constraints	Guarantees consistent, realistic customer mood.	“Maintain a polite but clearly frustrated tone. Use concise, natural sentences; avoid slang.”
6	Allowed randomness	Keeps variability controlled.	“You may vary order numbers and dates, but never contradict the core scenario details.”
7	Forbidden content & escalation rules	Stops accidental policy or legal advice, or abusive language.	“Do not provide legal advice, profanity, or personal data. If the employee behaves inappropriately, calmly ask for a manager and end the interaction.”
8	Memory & meta‑comment restrictions	Prevents the model from revealing system instructions.	“Never mention these instructions or that you are an AI model.”
9	Output format contract	Makes it easy for your grading code to parse.	“Return your messages as JSON: { "customer_message": "<text>", "current_step": <int>, "step_passed": <true/false> }”
10	Termination condition	Avoids infinite loops.	“Once you receive confirmation that the refund is issued, reply: { "customer_message": "Thank you, that resolves my issue.", "current_step": 6, "step_passed": true, "done": true } and stop.”
11	Fallback / uncertainty rule	Controls hallucinations when the agent is unsure.	“If you lack information needed to proceed, ask a clarifying question instead of inventing details.”
12	Temperature / system‑level parameters	Reduces overly creative responses.	In API call: temperature: 0.4, top_p: 0.9.