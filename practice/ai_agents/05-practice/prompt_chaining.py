import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from openai import OpenAI

# load API key
load_dotenv()

# Initialize the Gemini Client
# The Gemini API Key, which must be in your .env file as GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Define the base URL for the Gemini OpenAI compatibility layer
# This is the crucial step to redirect the openai client to the Gemini service.
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
class State(TypedDict):
    product_name: str
    basic_description: str
    features_benefits: str
    marketing_message: str
    final_description: str


# Define nodes (functions)
def generate_basic_description(state: State):
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": "Write a short, clear product description."},
            {"role": "user", "content": f"describe the product {state['product_name']}"}
        ]
    )

    # Returns key-value pair that langgraph merge with the global STATE object.
    return {"basic_description": response.choices[0].message.content}


def add_features_benefits(state: State):
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "user", "content": f"list features and benefits of {state['basic_description']}"}
        ]
    )

    return {"features_benefits": response.choices[0].message.content}


def create_marketing_message(state: State):
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "user", "content": f"Create a marketing message based on {state['features_benefits']}"}
        ]
    )

    return {"marketing_message": response.choices[0].message.content}


def polish_final_description(state: State):
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "user", "content": f"combine and polish into a final description using this marketing message {state['marketing_message']}"}
        ]
    )

    return {"final_description": response.choices[0].message.content}


# Define langgraph workflow with nodes and edges.
def build_workflow():

    # Initializes a langgraph workflow where each node operates on the state.
    wf = StateGraph(State)

    # Add the nodes to the langgraph.
    wf.add_node("basic", generate_basic_description)
    wf.add_node("features", add_features_benefits)
    wf.add_node("marketing", create_marketing_message)
    wf.add_node("final", polish_final_description)

    # Add the edges to the langgraph.
    wf.add_edge(START, "basic")
    wf.add_edge("basic", "features")
    wf.add_edge("features", "marketing")
    wf.add_edge("marketing", "final")
    wf.add_edge("final", END)

    # Compile the graph.
    return wf.compile()


# Run the workflow
if __name__ == "__main__":

    product_name = "Smart reusable notebook with cloud sync"

    # The state object is initialized with all empty states except the product_name.
    # These state names must exactly be same as in the State class (see line 32 above).
    state = State(
        product_name=product_name,
        basic_description="",
        features_benefits="",
        marketing_message="",
        final_description=""
    )

    # Build and execute the workflow.
    graph_wf = build_workflow()
    result = graph_wf.invoke(state)

    # Output results
    print("### Basic Description")
    print(result["basic_description"])
    print("=====================================================")
    print("### Features and Benefits")
    print(result["features_benefits"])
    print("=====================================================")
    print("### Marketing Message")
    print(result["marketing_message"])
    print("=====================================================")
    print("### Final Description")
    print(result["final_description"])
    print("=====================================================")
