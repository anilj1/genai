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
    topic: str
    advertisement: str
    review: str
    tagline: str
    final_output: str


# Define nodes (functions)
def generate_advertisement(state: State):
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": "You are a copywriter."},
            {"role": "user", "content": f"write a short advertisement for a {state['topic']}"}
        ]
    )

    # Returns key-value pair that langgraph merge with the global STATE object.
    return {"advertisement": response.choices[0].message.content}


def generate_review(state: State):
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": "You are a product reviewer."},
            {"role": "user", "content": f"write a short product review for a {state['topic']}"}
        ]
    )

    return {"review": response.choices[0].message.content}


def generate_tagline(state: State):
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": "You are a brand strategist."},
            {"role": "user", "content": f"write a catchy tagline for a {state['topic']}"}
        ]
    )

    return {"tagline": response.choices[0].message.content}


def combine_output(state: State):
    combined = f""""{state['advertisement']}\n\n{state['review']}\n\n{state['tagline']}"""
    return {"final_output": combined}


# Define langgraph workflow with nodes and edges.
def build_workflow():
    # Initializes a langgraph workflow where each node operates on the state.
    wf = StateGraph(State)

    # Add the nodes to the langgraph.
    wf.add_node("advertisement", generate_advertisement)
    wf.add_node("review", generate_review)
    wf.add_node("tagline", generate_tagline)
    wf.add_node("merge", combine_output)

    # Add the parallel edges to the langgraph.
    wf.add_edge(START, "advertisement")
    wf.add_edge(START, "review")
    wf.add_edge(START, "tagline")

    # Add the output edges to the langgraph.
    wf.add_edge("advertisement", "merge")
    wf.add_edge("review", "merge")
    wf.add_edge("tagline", "merge")
    wf.add_edge("merge", END)

    # Compile the graph.
    return wf.compile()


# Run the workflow
if __name__ == "__main__":
    # Input your product
    topic_name = "Apple Macbook Pro"

    # The state object is initialized with all empty states except the product_name.
    # These state names must exactly be same as in the State class (see line 32 above).
    state = State(
        topic=topic_name,
        advertisement="",
        review="",
        tagline="",
        final_output="",
    )

    # Build and execute the workflow.
    graph_wf = build_workflow()
    result = graph_wf.invoke(state)

    # Output results
    print("## Generated Product advertisement")
    print("### Advertisement")
    print(result["advertisement"])
    print("=====================================================")
    print("### Product review")
    print(result["review"])
    print("=====================================================")
    print("### Marketing Tagline")
    print(result["tagline"])
    print("=====================================================")
    print("### Final Output")
    print(result["final_output"])
    print("=====================================================")
