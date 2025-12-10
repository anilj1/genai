import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI

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


# Define the shared state
class DeptRouterState(TypedDict):
    inquiry: str
    department: str
    response: str


# Define the agent nodes
def sales_agent(state: DeptRouterState) -> dict:
    # Note that the agent currently returns a fixed response. Once we make this LLM capable,
    # They can return a more descriptive response.
    # This LLM can actually be a RAG based agent, which will return even more specific response.
    state["response"] = f"Sales team: we will help you with your sales request: {state['inquiry']}"
    return state


def support_agent(state: DeptRouterState) -> dict:
    state["response"] = f"Support team: we will help you with your support request: {state['inquiry']}"
    return state


def hr_agent(state: DeptRouterState) -> dict:
    state["response"] = f"HR team: we will help you with your hr request: {state['inquiry']}"
    return state


# Create a LLM router - ***FIXED*** to return 'state'
def router_agent(state: DeptRouterState) -> dict:
    # 1. System Instruction: The rules for the model.
    system_instruction = """
    You are an expert department router.
    Analyze the user's inquiry and decide whether it should be routed to 'sales', 'support', or 'hr'.
    You must respond with ONLY ONE WORD: sales, support, or hr. Do not include any other text or punctuation.
    """

    # 2. User Input: The actual inquiry.
    user_inquiry = state['inquiry']

    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            # System role sets the context and response format
            {"role": "system", "content": system_instruction},
            # User role provides the content to be classified
            {"role": "user", "content": user_inquiry},
        ]
    )

    decision = response.choices[0].message.content.strip().lower()

    # Update the state with the decision
    state["department"] = decision

    # A node MUST return the updated state dictionary
    return state


# Build the routing graph - ***FIXED*** conditional edge
def build_routing_graph():
    graph = StateGraph(DeptRouterState)
    graph.add_node("router", router_agent)
    graph.add_node("sales_agent", sales_agent)
    graph.add_node("support_agent", support_agent)
    graph.add_node("hr_agent", hr_agent)
    # The 'end' node isn't strictly necessary since END is used for the final edges, but kept for clarity
    # graph.add_node("end", lambda s: s) # Removed 'end' node as it's not strictly needed when using END

    # Set the entry point
    graph.set_entry_point("router")

    # Add conditional edges
    graph.add_conditional_edges(
        "router",
        # Use a lambda function to read the 'department' key from the state
        # that the 'router' node just updated and returned.
        lambda state: state["department"],  # <--- THIS IS THE SECONDARY FIX
        {
            "sales": "sales_agent",
            "support": "support_agent",
            "hr": "hr_agent"
        }
    )

    # Add edges to END
    graph.add_edge("sales_agent", END)
    graph.add_edge("support_agent", END)
    graph.add_edge("hr_agent", END)

    # Compile the graph.
    return graph.compile()


# Run the workflow
if __name__ == "__main__":
    # The state object is initialized with all empty states except the inquiry.
    state = DeptRouterState(
        inquiry="Can you tell me how many leaves are remaining?",
        department="",
        response=""
    )

    # Build and execute the workflow.
    graph_wf = build_routing_graph()
    result = graph_wf.invoke(state)

    # Output results
    print("## Department name")
    print(result["department"])
    print("=====================================================")

    print("## Response data")
    print(result["response"])
    print("=====================================================")