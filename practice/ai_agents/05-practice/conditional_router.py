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
    genre: str
    story: str


# Define nodes (functions)
def generate_horror_story(state: State):
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": "You are a horror story writer."},
            {"role": "user", "content": f"write a short horror story about {state['topic']}"}
        ]
    )

    # Returns key-value pair that langgraph merge with the global STATE object.
    return {"story": response.choices[0].message.content}


def generate_sci_fi_story(state: State):
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": "You are a sci-fi story writer."},
            {"role": "user", "content": f"write a short science fiction story about {state['topic']}"}
        ]
    )

    return {"story": response.choices[0].message.content}


def generate_comedy_story(state: State):
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": "You are a comedy story writer."},
            {"role": "user", "content": f"write a short comedy story about {state['topic']}"}
        ]
    )

    return {"story": response.choices[0].message.content}


# Router function
def genre_router(state: State):
    genre = state['genre'].lower()

    if genre == "horror":
        return "horror_story"
    elif genre == "sci-fi":
        return "sci_fi_story"
    elif genre == "comedy":
        return "comedy_story"
    else:
        raise ValueError("unknown_genre")


# Define langgraph workflow with nodes and edges.
def build_story_router_workflow():

    # Initializes a langgraph workflow where each node operates on the state.
    wf = StateGraph(State)

    # Add the nodes to the langgraph.
    wf.add_node("horror_story", generate_horror_story)
    wf.add_node("sci_fi_story", generate_sci_fi_story)
    wf.add_node("comedy_story", generate_comedy_story)

    # Use the genre_router function itself as the condition/mapping.
    # This tells the graph to execute genre_router, and whatever string
    # it returns (e.g., "horror_story") is the name of the next node.
    wf.add_conditional_edges(
        START,  # Start from the beginning
        genre_router,  # The function that decides the next node
        {
            "horror_story": "horror_story",
            "sci_fi_story": "sci_fi_story",
            "comedy_story": "comedy_story",
        }
    )

    # Add the edges to the langgraph.
    wf.add_edge("horror_story", END)
    wf.add_edge("sci_fi_story", END)
    wf.add_edge("comedy_story", END)

    # Compile the graph.
    return wf.compile()


# Run the workflow
if __name__ == "__main__":

    # The state object is initialized with all empty states except the product_name.
    # These state names must exactly be same as in the State class (see line 32 above).
    state = State(
        topic="An AI that goes rogue in space",
        genre="sci-fi",
        story=""
    )

    # Build and execute the workflow.
    graph_wf = build_story_router_workflow()
    result = graph_wf.invoke(state)

    # Output results
    print("## Generated Story")
    print(result["story"])
    print("=====================================================")
