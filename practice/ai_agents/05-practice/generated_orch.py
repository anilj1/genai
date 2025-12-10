import os
from typing import TypedDict, List, Dict, Union
from langgraph.graph import StateGraph, END
# --- CHANGE 1: Import the correct LLM class for Google/Gemini ---
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from dotenv import load_dotenv


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


# --- IMPORTANT SETUP ---
# Ensure you have the library installed: pip install langgraph langchain-google-genai pydantic
# Ensure your API key is set: export GEMINI_API_KEY='YOUR_API_KEY'

# --- 1. State Definition ---
# Define the State as a TypedDict for type safety and shared context
class EnquiryState(TypedDict):
    """Represents the state of the client enquiry as it moves through the graph."""
    enquiry_id: str
    raw_enquiry: str
    processed_data: Dict[str, str]
    evaluation_score: float
    required_task: str
    task_output: str
    agent_history: List[str]


# --- 2. Configuration ---
# --- CHANGE 2: Initialize the LLM using ChatGoogleGenerativeAI ---
llm = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL
)


# --- 3. Node/Agent Implementations ---

def process_data_node(state: EnquiryState) -> Dict[str, Union[Dict[str, str], List[str]]]:
    """
    Node 1: Data Pre-processor (Simulated Entity Extraction)
    Extracts key entities from the raw enquiry.
    """
    raw_enquiry = state["raw_enquiry"]

    # Simple extraction logic (can be replaced by a structured extraction LLM call)
    processed_data = {
        "client_name": "John Doe" if "John" in raw_enquiry else "Client",
        "contact_email": "client@example.com",
        "product_interest": "Premium Service" if "premium" in raw_enquiry.lower() else "Standard Service"
    }

    print(f"✅ Processed Data: Extracted {processed_data['product_interest']} interest.")

    return {
        "processed_data": processed_data,
        "agent_history": state.get("agent_history", []) + ["Data Processor"]
    }


def evaluator_node(state: EnquiryState) -> Dict[str, Union[float, List[str]]]:
    """
    Node 2: Evaluation Agent (Simulated Scoring)
    Scores the enquiry based on complexity/urgency using an LLM.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an AI sales evaluator. Analyze the client enquiry for complexity and urgency. Assign a score between 0.0 (simple/FAQ) and 1.0 (complex/urgent project)."),
        ("human",
         f"Enquiry: {state['raw_enquiry']}\n\nRespond ONLY with the score as a floating-point number, e.g., '0.75'.")
    ])

    chain = prompt | llm

    # Use a try/except block for robustness in parsing LLM output
    try:
        response = chain.invoke({})
        # Attempt to find a float in the response content
        score_text = response.content.split()[0].strip()
        score = float(score_text)
        # Ensure score is within valid range
        score = max(0.0, min(1.0, score))
    except Exception as e:
        print(f"⚠️ Evaluator LLM failed to parse (Error: {e}), defaulting to 0.5.")
        score = 0.5

    print(f"✅ Evaluation Score: {score:.2f} (0.0=Simple, 1.0=Urgent)")

    return {
        "evaluation_score": score,
        "agent_history": state.get("agent_history", []) + ["Evaluator"]
    }


def orchestrator_node(state: EnquiryState) -> Dict[str, str]:
    """
    Node 3: Orchestrator Agent (Intelligent Routing Decision)
    Determines the next required task based on the evaluation score.
    """
    score = state["evaluation_score"]

    # Routing Logic
    if score >= 0.8:
        task = "Schedule Expert Call"
    elif score >= 0.4:
        task = "Generate Standard Quote"
    else:
        task = "Auto-Reply from KB"  # This task leads to the END state

    print(f"✅ Orchestrator Decision: Task set to '{task}'")

    return {
        "required_task": task,
        "agent_history": state.get("agent_history", []) + ["Orchestrator"]
    }


def task_agent_node(state: EnquiryState) -> Dict[str, str]:
    """
    Node 4: Task-Specific Agent (Execution of the Determined Task)
    Simulates executing a tool/API call based on the required_task.
    """
    task = state["required_task"]

    if task == "Schedule Expert Call":
        output = f"Booking link sent to {state['processed_data']['contact_email']}. Expert assigned: Dr. Smith."
    elif task == "Generate Standard Quote":
        output = (f"Generated standard quote for {state['processed_data']['product_interest']}. "
                  f"Price: $5,000. Quote sent to CRM system.")
    else:
        output = "Error: Unknown task assigned."

    print(f"✅ Task Agent Execution: Completed '{task}'.")

    return {
        "task_output": output,
        "agent_history": state.get("agent_history", []) + ["Task Agent"]
    }


# --- 4. Routing Logic (Conditional Edge) ---

def route_next_step(state: EnquiryState) -> str:
    """Routes the flow based on the task set by the Orchestrator."""
    task = state["required_task"]

    if task in ["Schedule Expert Call", "Generate Standard Quote"]:
        return "continue_to_task_agent"

    # If task is "Auto-Reply from KB", we terminate the workflow
    print("➡️ Routing: Simple enquiry, ending workflow with auto-reply.")
    return "end_workflow"


# --- 5. Graph Construction and Compilation ---

def create_enquiry_workflow():
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(EnquiryState)

    # 1. Add Nodes
    workflow.add_node("process_data", process_data_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("task_agent", task_agent_node)

    # 2. Set Entry Point
    workflow.set_entry_point("process_data")

    # 3. Add Edges (Linear Flow)
    workflow.add_edge("process_data", "evaluator")
    workflow.add_edge("evaluator", "orchestrator")

    # 4. Add Conditional Edge (Orchestration Logic)
    workflow.add_conditional_edges(
        "orchestrator",
        route_next_step,
        {
            "continue_to_task_agent": "task_agent",
            "end_workflow": END
        }
    )

    # 5. Connect Task Agent to the end
    workflow.add_edge("task_agent", END)

    # Compile the graph
    app = workflow.compile()
    print("--- LangGraph Workflow Compiled Successfully ---")
    return app


# --- 6. Execution ---

if __name__ == "__main__":
    # --- CHANGE 3: Check for GEMINI_API_KEY environment variable ---
    if "GEMINI_API_KEY" not in os.environ:
        print("FATAL ERROR: Please set the GEMINI_API_KEY environment variable.")
    else:
        app = create_enquiry_workflow()

        # --- Run Scenario 1: Complex/Urgent Enquiry ---
        print("\n\n=== SCENARIO 1: COMPLEX ENQUIRY (Expected: Schedule Expert Call) ===")
        initial_state_1 = EnquiryState(
            enquiry_id="E001",
            raw_enquiry="I have a complex multi-year project requiring custom AI integration. Need to discuss premium service and timelines urgently.",
            processed_data={},
            evaluation_score=0.0,
            required_task="",
            task_output="",
            agent_history=[]
        )

        final_state_1 = app.invoke(initial_state_1)
        print("\n--- FINAL STATE SUMMARY 1 ---")
        print(f"Path Taken: {' -> '.join(final_state_1['agent_history'])}")
        print(f"Final Task: {final_state_1['required_task']}")
        print(f"Result: {final_state_1['task_output']}")

        print("\n" + "=" * 70)

        # --- Run Scenario 2: Simple/FAQ Enquiry ---
        print("\n\n=== SCENARIO 2: SIMPLE ENQUIRY (Expected: Auto-Reply, END) ===")
        initial_state_2 = EnquiryState(
            enquiry_id="E002",
            raw_enquiry="What are your standard business hours and do you offer free trials?",
            processed_data={},
            evaluation_score=0.0,
            required_task="",
            task_output="",
            agent_history=[]
        )

        final_state_2 = app.invoke(initial_state_2)
        print("\n--- FINAL STATE SUMMARY 2 ---")
        print(f"Path Taken: {' -> '.join(final_state_2['agent_history'])}")
        print(f"Final Task: {final_state_2['required_task']}")
        # The task agent isn't run, so the required_task remains set by the orchestrator before termination
        print(f"Result: Workflow ended with Auto-Reply decision (Task: {final_state_2['required_task']}).")