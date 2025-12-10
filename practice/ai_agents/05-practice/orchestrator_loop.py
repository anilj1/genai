import datetime
import os
from typing import List, Annotated
from operator import add

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI
import matplotlib.pyplot as plt
import networkx as nx

# load API key
load_dotenv()

# Initialize the Gemini Client configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Define the base URL for the Gemini OpenAI compatibility layer
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Use a supported Gemini model name for the compatibility layer
GEMINI_MODEL = "gemini-2.5-flash"

# Initialize the OpenAI client configured for the Gemini API
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not found. Please check your .env file.")

# Initialize the OpenAI client with the Gemini base URL and API key
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL
)


# --- State Definition using TypedDict ---
class InquiryState(TypedDict):
    """ Tracks the inquiry lifecycle state using TypedDict for reliable updates. """
    client_name: str
    client_email: str
    request_details: str
    is_approved: bool
    evaluation_notes: str
    appointment_time: str
    crm_log: str
    activity_log: Annotated[List[str], add]
    evaluation_attempts: int


def get_initial_state(client_name: str, client_email: str, request_details: str) -> dict:
    """ Returns a dictionary representing the initial state with default values. """
    return {
        "client_name": client_name,
        "client_email": client_email,
        "request_details": request_details,
        "is_approved": False,
        "evaluation_notes": "",
        "appointment_time": "",
        "crm_log": "",
        "activity_log": [],
        "evaluation_attempts": 0,
    }


# Intake agent
def intake_agent(state: InquiryState) -> dict:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updates = {}
    updates["activity_log"] = [
        f"Request received from **{state['client_name']}** ({state['client_email']}) at {timestamp}."]

    if not state.get("is_approved", True) and state.get("evaluation_notes") and "Denied" in state["evaluation_notes"]:
        print("\n--- DENIAL LOOP TRIGGERED ---")
        print(f"Previous denial reason: {state['evaluation_notes']}")

        current_request = state["request_details"].lower()

        if "production database" in current_request and state["evaluation_attempts"] == 1:
            new_request = "Can I have **read-only** access to the **staging** database?"
            updates["request_details"] = new_request
            updates["activity_log"].append(f"**Intake Rerun:** Request modified to: '{new_request}'")
        else:
            updates["activity_log"].append(f"**Intake Rerun:** Request re-submitted without modification.")

    return updates


def evaluation_agent(state: InquiryState) -> dict:
    updates = {}
    new_attempts = state.get("evaluation_attempts", 0) + 1
    updates["evaluation_attempts"] = new_attempts

    try:
        system_instruction = f"Approve or deny the following client request. Respond only with 'approved' or 'denied'. Deny requests for 'production' access or 'admin' rights."

        response = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Client request: {state['request_details']}"}
            ],
            temperature=0.2,
            max_tokens=200
        )

        decision = response.choices[0].message.content.strip().lower()
        request_details_lower = state['request_details'].lower()

        # Fix: Ensure non-sensitive requests are reliably approved
        if "denied" in decision or "production" in request_details_lower or "admin" in request_details_lower:
            updates["is_approved"] = False
            if "production database" in request_details_lower:
                updates["evaluation_notes"] = "Denied. Access to production systems is restricted."
            elif "admin" in request_details_lower:
                updates["evaluation_notes"] = "Denied. Admin rights are restricted."
            else:
                updates["evaluation_notes"] = "Denied."
        else:
            updates["is_approved"] = True
            updates["evaluation_notes"] = "Approved."

        updates["activity_log"] = [f"Evaluation (Attempt {new_attempts}): " + updates["evaluation_notes"]]

    except Exception as e:
        updates["is_approved"] = False
        updates["evaluation_notes"] = f"Evaluation failed: {e}"
        updates["activity_log"] = ["Evaluation error"]

    return updates


# --- NEW DEBUG AGENT ---
def debug_state_agent(state: InquiryState) -> dict:
    print("\n--- DEBUG: STATE AFTER EVALUATION ---")
    print(f"Request: {state['request_details']}")
    print(f"Is Approved (pre-router check): {state['is_approved']}")
    print(f"Evaluation Notes: {state['evaluation_notes']}")
    print(f"Attempts: {state['evaluation_attempts']}")
    print("------------------------------------\n")
    # Returns an empty dict as it only inspects the state, not modifies it
    return {}


# -----------------------

def scheduling_agent(state: InquiryState) -> dict:
    updates = {}
    if state["is_approved"]:
        meeting_time = datetime.datetime.now() + datetime.timedelta(days=1)
        updates["appointment_time"] = meeting_time.strftime("%Y-%m-%d %H:%M:%S")
        updates["activity_log"] = [f"Meeting scheduled for {updates['appointment_time']}"]
    return updates


def crm_update_agent(state: InquiryState) -> dict:
    updates = {}
    updates["crm_log"] = "CRM updated with inquiry status."
    updates["activity_log"] = ["CRM update completed."]
    return updates


# Router function
def route_evaluation(state: InquiryState) -> str:
    MAX_RETRIES = 2

    if state["is_approved"]:
        return "approved"
    else:
        if state["evaluation_attempts"] >= MAX_RETRIES:
            # We add a log here to ensure termination reason is captured before END
            state["activity_log"].append("Routing: Request denied. Max retries reached. Flow terminated.")
            return "terminate"
        else:
            state["activity_log"].append(
                f"Routing: Request denied, looping back to Intake (Retry {state['evaluation_attempts']}).")
            return "denied"


# Build the routing graph
def build_graph():
    graph = StateGraph(InquiryState)
    graph.add_node("intake", intake_agent)
    graph.add_node("evaluating", evaluation_agent)
    graph.add_node("debug_check", debug_state_agent)  # <-- New Node
    graph.add_node("scheduling", scheduling_agent)
    graph.add_node("crm_update", crm_update_agent)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "evaluating")

    # New Edge: Evaluating -> Debug Node
    graph.add_edge("evaluating", "debug_check")

    # Router source is now the debug node
    graph.add_conditional_edges(
        "debug_check",
        route_evaluation,
        {
            "approved": "scheduling",
            "denied": "intake",
            "terminate": END
        }
    )

    graph.add_edge("scheduling", "crm_update")
    graph.add_edge("crm_update", END)

    return graph.compile()


# Run the workflow
if __name__ == "__main__":

    # --- Test 1: Approved Request (Quota Upgrade) ---
    print("=" * 50)
    print("--- Test 1: Approved Request (Quota Upgrade) ---")
    print("=" * 50)

    initial_state_1 = get_initial_state(
        client_name="Sarah Connor",
        client_email="sarah@skynet.com",
        request_details="Can we upgrade our team's quota to 10M requests/month?"
    )

    graph = build_graph()
    result_1 = graph.invoke(initial_state_1)

    print("\n✅ Final output (Approved)")
    print(f"Request: {result_1['request_details']}")
    print(f"Is Approved: {result_1['is_approved']}")
    print("\nActivity Log:")
    for line in result_1['activity_log']:
        print(f"* {line}")