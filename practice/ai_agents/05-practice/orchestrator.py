import datetime
import os
from dataclasses import field, dataclass, asdict
from typing import List

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

# load API key
load_dotenv()

# Initialize the Gemini Client
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


# Shared state
@dataclass
class InquiryState:
    """ Tracks the inquiry lifecycle state """
    """ Passed around from each node of the graph with each agent """
    client_name: str            # User input
    client_email: str
    request_details: str
    is_approved: bool = False   # Once the inquiry is received, approved or not.
    evaluation_notes: str = ""  # Short note by LLM why request approved / denied.
    appointment_time: str = ""  #
    crm_log: str = ""           # Filled in by the CRM handler.
    activity_log: List[str] = field(default_factory=list)   # Running log of the flow execution.


# Intake agent
def intake_agent(state: InquiryState) -> dict:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state.activity_log.append(f"Request received from {state.client_name} ({state.client_email}) at {timestamp}.")
    return asdict(state)


def evaluation_agent(state: InquiryState) -> dict:
    try:
        system_instruction = f"Approve or deny the following client request. Respond only with 'approved' or 'denied'."

        response = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                # System role sets the context and response format
                {"role": "system", "content": system_instruction},
                # User role provides the content to be classified
                {"role": "user", "content": f"Client request: {state.request_details}"}
            ],
            temperature=0.2,
            max_tokens=200
        )

        decision = response.choices[0].message.content.strip().lower()
        state.is_approved = "approved" in decision
        state.evaluation_notes = "Approved." if state.is_approved else "Denied."
        state.activity_log.append("Evaluation: " + state.evaluation_notes)
    except Exception as e:
        state.is_approved = False
        state.evaluation_notes = f"Evaluation failed: {e}"
        state.activity_log.append("Evaluation error")

    return asdict(state)


def scheduling_agent(state: InquiryState) -> dict:
    if state.is_approved:
        meeting_time = datetime.datetime.now() + datetime.timedelta(days=1)
        state.appointment_time = meeting_time.strftime("%Y-%m-%d %H:%M:%S")
        state.activity_log.append(f"Meeting scheduled for {state.appointment_time}")


def crm_update_agent(state: InquiryState) -> dict:
    state.crm_log = "CRM updated with inquiry status."
    state.activity_log.append("CRM update completed.")
    return asdict(state)


# Build the routing graph
def build_graph():
    graph = StateGraph(InquiryState)
    graph.add_node("intake", intake_agent)
    graph.add_node("evaluating", evaluation_agent)
    graph.add_node("scheduling", scheduling_agent)
    graph.add_node("crm_update", crm_update_agent)
    graph.add_node("end", lambda state: asdict(state))

    # Set the entry point
    graph.set_entry_point("intake")
    graph.add_edge("intake", "evaluating")
    graph.add_edge("evaluating", "scheduling")
    graph.add_edge("scheduling", "crm_update")
    graph.add_edge("crm_update", "end")
    graph.set_finish_point("end")

    # Compile the graph.
    return graph.compile()


def visualize_grpah():
    graph = nx.DiGraph()
    graph.add_edges_from({
        ("START", "intake"),
        ("intake", "evaluation"),
        ("evaluation", "scheduling"),
        ("scheduling", "crm_update"),
        ("crm_update", "end"),
        ("end", "END")
    })

    plt.figure(figsize=(9, 5))
    nx.draw(graph, with_labels=True, node_size=2000, node_color="lightblue", font_size=12)
    path = "workflow.png"
    plt.savefig(path)
    return path


# Run the workflow
if __name__ == "__main__":
    # The state object with the inquiry details.
    initial_state = InquiryState(
        client_name="Peter Bob",
        client_email="peter@bob.com",
        request_details="Can we upgrade our team's quota to 10M requests/month?"
    )

    graph = build_graph()
    result = graph.invoke(asdict(initial_state))

    print("Final output")
    print(result)

    for line in result['activity_log']:
        print(line)

    print("===========================================")
    
    # Access to production database
    initial_state = InquiryState(
        client_name="Peter Bob",
        client_email="peter@bob.com",
        request_details="Can I have access to the production database?"
    )

    graph = build_graph()
    result = graph.invoke(asdict(initial_state))

    print("Final output")
    print(result)

    for line in result['activity_log']:
        print(line)

    # visualize_grpah()
