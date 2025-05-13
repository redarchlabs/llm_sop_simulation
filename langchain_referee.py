"""
return_simulation.py
-----------------------------------------------------------
Implements a three‑tier role‑play pipeline in LangGraph:

user UI  ─▶  Grader LLM  ─▶  Referee LLM  ─▶  Orchestrator
-----------------------------------------------------------
• Grader  = grades the user against the SOP.
• Referee = audits the Grader's judgement.
• Orchestrator = loops until Grader + Referee agree, then
                 advances to the next coach step.
-----------------------------------------------------------
Prereqs
-----------------------------------------------------------
pip install  langchain langgraph openai tiktoken pydantic
-----------------------------------------------------------
Set OPENAI_API_KEY in your environment before running.
Swap `ChatOpenAI` for `Ollama` or any other LC LLM if needed.
"""

from __future__ import annotations
from typing import Dict, Any, Literal, Optional, List, TypedDict
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
import openai
import dotenv
import os
from dotenv import load_dotenv
load_dotenv()
openai.organization = os.getenv("OPENAI_ORG_ID")
import json 

# =====================  CONFIG  ===========================
LLM_GRADER  = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
LLM_REFEREE = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
LLM_coach = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)


MAX_GRADER_RETRIES = 2

SOP_STEPS = {

}

# ================  PROMPT TEMPLATES  ======================

GRADER_GRADING_PROMPT = """\
You are the *Grader* LLM grading an user's reply in simulated scenario.
Return ONLY JSON:

{{
  "role": "grader",
  "message": "<short constructive feedback to user>",
  "current_step": {step},
  "step_passed": true | false
}}

Rules:
1. Use the SOP below to decide if the user satisfied step {step}.
2. Be concise (<40 tokens in "message").
3. No policy leaks, no profanity, no AI mentions.
4. JSON only, no markdown fences.

──────── SOP Step Details ────────:
{SOP_STEP_DETAILS}

"""


GRADER_INTERACTION_PROMPT = """\
You are the *Grader* LLM grading an user's reply in a simulated scenario for an SOP.
──────── FULL SOP Details ────────:
{SOP}

──────── CURRENT SOP Step Details ────────:
{SOP_STEP_DETAILS}

──────── Conversation History ────────:
Here is the conversation so far.  Please respond to the next question from the user.
{conversation_history}

──────── RULES ────────:
1. Do not tell the user exacly what to say/do without giving them a chance to explore.  Give them a hint if they ask for it, but do not give them the exact answer immediately.
2. After a few back and forths, you can give them the exact answer if they are still struggling.

"""

REFEREE_SYSTEM_PROMPT = """\
You are the **Referee** LLM.  
Goal: judge whether the *Grader* graded the user's reply **correctly** for SOP step {step}.

──────── Context you will receive (in the USER message) ────────
{{
  "user_reply": "<text>",
  "grader_feedback": {{            # JSON from the Grader
     "step_passed": true|false,
     ...
  }}
}}

──────── How to decide ─────────
1. Determine **user_passed** for step {step}.  
   • Use the literal rubric below.  
2. Compare it to **grader_feedback["step_passed"]**.  
3. Set **referee_grade**:  
   • `"pass"`  → grader correctly assessed the situation(i.e. the grader failed the user when it should have or passed when it should have failed).  
   • `"fail"`  → grader assessment is wrong.  
4. Set **must_regenerate**: `true` if referee_grade == "fail", else `false`.  
5. Write ONE short sentence of feedback to the grader on how well it did or did not evaluate. (≤ 40 tokens, no markdown).

──────── Rubric for step {step} — {step_desc} ────────
{SOP_STEP_DETAILS}

• No profanity or hostility.  
→ If both present → user_passed = true.  
→ Otherwise → user_passed = false.

──────── Special Handling for Grader Fails ────────

If the Grader flagged the step as not passed (step_passed == false)
AND you also believe the user did NOT meet the requirements for this step:

➡️ Then:
  • "referee_grade": "pass"
  • "must_regenerate": false

Do not mark the grader as wrong just because of wording or lack of explanation.
Your job is to validate correctness, not critique style.

──────── Output (JSON only, NO markdown) ─────────
{{
  "role": "referee",
  "current_step": {step},
  "message": "<referee feedback to grader on how the grader did grading, not on whether the user passed>",
  "must_regenerate": true | false,
  "referee_grade": "pass" | "fail"
}}
Be completely deterministic (temperature 0).

"""

# ================  STATE MODEL  ===========================


class StateDict(TypedDict):
    current_step: int
    grader_retries: int
    done: bool
    last_grader: Optional[Dict[str, Any]]
    last_referee: Optional[Dict[str, Any]]
    history: List[Dict[str, Any]]
    input_message: Optional[Dict[str, str]]  # user reply
    coach_message: Optional[str]          # most recent coach reply
    dialogue_history: List[Dict[str, str]]   # alternating coach/user turns
    next_node: Optional[str]                # next node to run

# ================  NODE FUNCTIONS  ========================

def coach_node(state: StateDict) -> Dict[str, Any]:
    
    if state.get("next_node")  and state["next_node"] != "coach":
        # Skip this node if not the next one
        return {}
    
    dialogue = state.get("dialogue_history", [])
    if not dialogue:
        prompt = f"""
        <LLM INSTRUCTIONS>
        This is a simulation for a training system.  
        A user is being trained to follow a Standard Operating Procedure (SOP) for a specific task.

-----------------------SOP DETAILS------------------------
{SOP_STEPS}
----------------------------------------------------------

        You are simulating the role of the coach in the SOP training simulation.  You are not a grader or referee.  
        You only represent the coach being played out in the SOP.  Do not include any other roles or perspectives.
        Do not mention AI, LLM, or anything like that.
        You can provide hints to the user if they ask for it, but never give them the exact answer.  Give them a chance to explore and learn.
        If they are really struggling, you can give them the exact answer, but only after a few back and forths.
        You are evaulating the user's input and giving them feedback on how well they are doing.
        You are not grading the user.  You are not a referee.  You are not a grader.  You are just a coach.
        Do not respond as the user.  Do not respond as the grader.  Do not respond as the referee.
        Do not mention the grader or referee.  Do not mention the user.  Do not mention the coach.

        Go ahead and describe the training simulation situation to the user from what they need to know to start the simulation as the user.  What is the situation they are trying to 
        role play?  What is the task they are trying to accomplish?  What is the goal of the simulation?
        </LLM INSTRUCTIONS>
        """
        dialogue_text = ""
    else:
        dialogue_text = "\n".join([f"{x['role'].capitalize()}: {x['content']}" for x in dialogue])
        prompt = f"""
You are a coach in an Standard Operating procedure.

Here is the conversation so far:
{dialogue_text}

Respond naturally. Keep it short (1–2 sentences). Be polite but firm. Do not repeat yourself.
Don't respond as the user or the trainee. Only respond as if you are the coach.

        You are simulating the role of the coach in the SOP training simulation.  You are not a grader or referee.  
        You only represent the coach being played out in the SOP.  Do not include any other roles or perspectives.
        Do not mention AI, LLM, or anything like that.
        You can provide hints to the user if they ask for it, but never give them the exact answer.  Give them a chance to explore and learn.
        If they are really struggling, you can give them the exact answer.
        You are evaulating the user's input and giving them feedback on how well they are doing.
        You are not grading the user.  You are not a referee.  You are not a grader.  You are just a coach.
        Do not mention the grader or referee.  Do not mention the user.  Do not mention the coach.

"""

    result = LLM_coach([SystemMessage(content=prompt),HumanMessage(content=dialogue_text)])
    grader_reply = result.content.strip()
    print(f"\n🗣️  Coach:{grader_reply}")
    return {
        "coach_message": grader_reply,
        "input_message": {"role": "coach", "content": grader_reply},
        "last_grader": None,
        "last_referee": None,
        "history": state.get("history", []) + [{"role": "coach", "content": grader_reply}],
        "dialogue_history": dialogue + [{"role": "coach", "content": grader_reply}],
        "next_node": "grader"
    }



# Node functions now accept the state dictionary
def grader_node(state: StateDict) -> Dict[str, Any]:
    """Grader grades the current coach's reply."""
    grader_message_content = state.get("input_message", {}).get("content")
    direct_question_to_grader = grader_message_content.startswith("Grader:")
    if not direct_question_to_grader and state.get("next_node","") != "grader":
        # Skip this node if not the next one
        return {"next_node": state["next_node"]}

    
    # Ensure input_message exists and has content
    grader_message_content = state.get("input_message", {}).get("content")
    if not grader_message_content:
         # Handle case where there's no message to grade (shouldn't happen with driver loop)
         print("Error: Grader node received no input message.")
         return {"last_grader": {"role": "grader", "message": "No message to grade.", "current_step": state["current_step"], "step_passed": False}}

    #is this a direct question to the grader?  If so, it will not be graded.
    if direct_question_to_grader:
        print("Grader: This is a direct question to the grader.  It will not be graded.")
        
        prompt = PromptTemplate.from_template(GRADER_INTERACTION_PROMPT).format(
            SOP=SOP_STEPS,
            step=state["current_step"], # Access using dictionary keys
            SOP_STEP_DETAILS=SOP_STEPS[state["current_step"]],
            conversation_history=state["history"],
        )

        grader_response = LLM_GRADER([
            SystemMessage(content=prompt),
            HumanMessage(content=grader_message_content)
        ])
        # Access .content before parsing JSON
        last_grader = {
            "role": "grader",
            "message": grader_response.content,
            "current_step": state["current_step"],
            "step_passed": False
        }
        return {"last_grader": last_grader, "next_node":"user", "history": state["history"] + [last_grader]}

    else:
        print("Grader is grading the user's reply...")

        prompt = PromptTemplate.from_template(GRADER_GRADING_PROMPT).format(
            step=state["current_step"], # Access using dictionary keys
            step_desc=get_sop_step_description(state["current_step"]),
            SOP_STEP_DETAILS=SOP_STEPS[state["current_step"]]
        )

        grader_response = LLM_GRADER([
            SystemMessage(content=prompt),
            HumanMessage(content=grader_message_content)
        ])
        # Access .content before parsing JSON
        grader_json = JsonOutputParser().parse(grader_response.content)
        # Return dictionary update for the state
        print(grader_json)
        return {"last_grader": grader_json, "next_node":"referee"}


def referee_node(state: StateDict) -> Dict[str, Any]:
    """Referee audits the grader judgement."""

    if state.get("next_node")  and state["next_node"] != "referee":
        # Skip this node if not the next one
        return {"last_referee":None}

    # Access needed data from state dictionary
    print("Referee is grading grader's reply...")
    grader_message_content = state.get("input_message", {}).get("content")
    grader_feedback = state.get("last_grader")

    if not grader_message_content or not grader_feedback:
        # Handle missing data - graph shouldn't reach here if edges are correct
        print("Error: Referee node missing input message or grader feedback.")
        return {"last_referee": {"referee_grade": "fail", "feedback": "Missing inputs.", "must_regenerate": True}}


    system_prompt = PromptTemplate.from_template(REFEREE_SYSTEM_PROMPT).format(
        step=state["current_step"], # Access using dictionary keys
        step_desc=get_sop_step_description(state["current_step"]), # Access using dictionary keys
        SOP_STEP_DETAILS=SOP_STEPS[state["current_step"]]
    )
    referee_input = {
        "grader_reply": grader_message_content, # Get content from state
        "grader_feedback": grader_feedback      # Get last_grader from state
    }
    referee_response = LLM_REFEREE([
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(referee_input))
    ])
    # Access .content before parsing JSON
    referee_json = JsonOutputParser().parse(referee_response.content)

    # # catch inconsistency
    # if referee_json["referee_grade"] == "fail" and referee_json["must_regenerate"]:
    #     referee_json["referee_grade"] = "pass"

    # if referee_json["referee_grade"] == "fail" and referee_json["must_regenerate"]==False:
    #     referee_json["referee_grade"] = "pass"

    print(f"Grader JSON: {grader_feedback}")
    print(f"Referee JSON: {referee_json}")

    # Return dictionary update for the state
    return {"last_referee": referee_json}

def user_node(state: StateDict) -> Dict[str, Any]:
    print(f"\n📝 SOP Step {state['current_step']} — {get_sop_step_description(state['current_step'])}")
    input_text = input("\n👨‍💼 Your reply:\n> ")

    reply = {"role": "user", "content": input_text}
    next_node = "grader"
    if input_text.lower().startswith("coach:"):
        next_node = "coach"

    return {
        "input_message": reply,
        "history": state["history"] + [reply],
        "next_node": next_node
    }

def orchestrator_node(state: StateDict) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}

    if updates.get("done",False):
        updates["next_node"] = END
        return updates  # ✅ always return a dict
    #updates["next_node"] = state.get("next_node", "coach")
    referee = state.get("last_referee",None)
    grader = state.get("last_grader", None)
    input_msg = state.get("input_message", None)

    #possible states:
    #1.) User asked a direct question to the grader and it should not be graded.
    #2.) User responded to the coach in the dialogue and it should be graded.
    #3.) Simulation is done and the user passed all steps.
    #4.) Grader failed the user and the referee agreed.
    #5.) Grader failed the user and the referee disagreed.
    #6.) Grader passed the user and the referee disagreed.
    #7.) Grader passed the user and the referee agreed.


    # if referee is None or grader is None :
    #     print("❌ ERROR: Missing data in orchestrator")
    #     updates["next_node"] = "coach"
    #     return updates

    if referee:
        # Referee is present, so we can check the grading
        grader_passed = grader["step_passed"]
        referee_agreed = referee["referee_grade"] == "pass"

        if referee_agreed and grader_passed:
            # ✅ user passed the step
            updates["dialogue_history"] = state["dialogue_history"] + [input_msg]
            updates["grader_retries"] = 0
            updates["current_step"] = min(state["current_step"] + 1, 6)
            updates["done"] = state["current_step"] >= 6
            updates["input_message"] = None
            updates["last_grader"] = None
            updates["last_referee"] = None
            updates["next_node"] = END if updates["done"] else "coach"


        elif referee_agreed and not grader_passed:
            # ❌ Grader correctly failed the user
            updates["grader_retries"] = 0
            updates["input_message"] = None
            updates["next_node"] = "user"
            updates["history"] = state["history"] + [input_msg]  # Log failed attempt


        else:
            # 🚫 Referee says Grader misgraded
            retries = state.get("grader_retries", 0) + 1
            if retries == MAX_GRADER_RETRIES:
                raise RuntimeError("💥 Grader failed too many times.")
            updates["grader_retries"] = retries
            updates["next_node"] = "grader"

    return updates

# Initialize the StateGraph
graph = StateGraph(StateDict)

# Add all nodes
graph.add_node("coach", coach_node)
graph.add_node("user", user_node)
graph.add_node("grader", grader_node)
graph.add_node("referee", referee_node)
graph.add_node("orchestrator", orchestrator_node)

# Core transition edges
graph.add_edge("coach", "user")
graph.add_edge("user", "grader")
graph.add_edge("grader", "referee")
graph.add_edge("referee", "orchestrator")

graph.add_conditional_edges("orchestrator", {
    "coach": lambda state: state["next_node"] == "coach",
    "user": lambda state: state["next_node"] == "user",
    END: lambda state: state["next_node"] == END
})


# Set the entry point — where the whole loop starts
graph.set_entry_point("coach")

# Compile the graph
simulation = graph.compile()


# ================  DRIVER LOOP  ===========================

def run_simulation(sample: Dict[str, Any]) -> None:
    # Initialize state as a dictionary matching StateDict structure
    state: StateDict = {
        "current_step": 1,
        "grader_retries": 0,
        "done": False,
        "last_grader": None,
        "last_referee": None,
        "history": [],
        "dialogue_history": [],
        "input_message": None,
        "coach_message": None,
    }

    steps_array = sample.get("steps", {})
    for step in steps_array:
        interaction_number = step.get("step_number")
        SOP_STEPS[interaction_number] = step
        print(f"Step {interaction_number}: {get_sop_step_description(interaction_number)}")

    print("=== Retail Return Simulation ===")

    while not state["done"]:

        state = simulation.invoke(state)

        # # ---- Get input from the current coach (here, hardcoded as 'user') ----
        # # This part can be extended to get input from different coachs dynamically
        # current_step = state["current_step"]

        # # ───── Display dialogue history in chat format ─────
        # # if state.get("history"):
        # #     print("\n💬 Conversation so far:\n")
        # #     for turn in state["history"]:
        # #         role = turn["role"]
        # #         text = turn["content"]
        # #         prefix = "🗣️ coach:" if role == "coach" else "👨‍💼 user:"
        # #         print(f"{prefix:<13} {text}")
            
        

        # step_desc = SOP_STEPS.get(current_step, "Unknown Step")
        # input_text = input(
        #     f"\n👨‍💼  user reply for step {current_step} "
        #     f"({step_desc}):\n> "
        # )

        # # ---- Populate the input_message key in the state dictionary ----
        # # This structure allows specifying the coach's role
        # state["input_message"] = {"role": "user", "content": input_text}

        # # ---- run a graph “tick” ----
        # # Invoke with the state dictionary directly
        # try:
        #     # The state dictionary is updated in place by LangGraph
        #     state = simulation.invoke(state)
        # except RuntimeError as e:
        #     print(f"\n❌ Simulation failed: {e}")
        #     break # Exit loop on error

        # ---- Show feedback based on the updated state ----
        # Access feedback from the state dictionary
        last_grader   = state.get("last_grader", None)
        last_referee = state.get("last_referee", None)
    
        if not last_referee:
            if last_grader:
                print(f"\n📝 Grader: {last_grader['message']}")
            continue
    
        if last_referee["referee_grade"] == "pass":
            # ✅ Referee agrees with Grader — show grader feedback
            print(f"\n📝 Grader: {last_grader['message']}")
            # Optional: also show Referee confirmation sentence
            # print(f"✅ Referee: {last_referee['message']}")
        else:
            # ❌ Referee disagrees — warn trainee
            print(f"⚠️  Referee disagreed: {last_referee['message']}")

    # After the loop (either done or error)
    if state["done"]:
         print("\n✅  Simulation complete! All steps passed.")
         

    # ---- Show Simulation History ----
    print("\n=== Simulation History ===")
    if not state["history"]:
        print("No steps completed successfully.")
    else:
        # History records steps that successfully advanced (referee referee_grade == "pass")
        # The index of the history entry corresponds to the step number (1-based)
        for i, entry in enumerate(state["history"]):
            interaction_number = i + 1
            print(f"{interaction_number}):")
            print(f"  Coach ({entry['role']}): \"{entry.get('content',entry.get('message',''))}\"") # Use role and content from history entry
            if entry.get("grader_feedback"):
                print(f"  Grader Feedback: \"{entry['grader_feedback']}\"") # Use grader feedback from history
            print("-" * 20) # Separator

    print("=== End Simulation History ===")

def get_sop_step_description(step_number):
    return SOP_STEPS.get(step_number, {"step_name": "Unknown step","step_number": 1,"rubric": {"description": "No description","example_message": ""  }}).get("rubric", {"description": "No description","example_message": ""  }).get("description", "No description")

if __name__ == "__main__":
    #load the sample file
    sample = None
    with open("sample_simulations/SOP145.json", "r") as f:
        sample = json.load(f)
    run_simulation(sample)