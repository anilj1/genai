import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
import serpapi

# load API key
load_dotenv()

# Initialize the Gemini Client
# The Gemini API Key, which must be in your .env file as GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

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
    query: str
    intent: str
    optimized_query: str
    search_results: str
    format_answer: str
    final_response: str


# Define nodes (functions)
def extract_intent(state: State):
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": "You are an intent extraction assistant."},
            {"role": "user", "content": f"What is user tying to do with this query? {state['query']}"}
        ]
    )

    # Returns key-value pair that langgraph merge with the global STATE object.
    return {"intent": response.choices[0].message.content.strip()}


def optimized_query(state: State):
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in optimizing web search queries. Your task is to "
                                          "rewrite the query based on the provided intent."},
            {"role": "user", "content": f"User Intent: {state['intent']}\n"
                                        "Rewrite the original query to better reflect the user intent."}
        ]
    )

    # Returns key-value pair that langgraph merge with the global STATE object.
    return {"optimized_query": response.choices[0].message.content.strip()}


def perform_search(state: State):
    params = {
        "engine": "google",
        "q": state['optimized_query'],
        "api_key": SERP_API_KEY
    }

    search = serpapi.search(params)

    # Extract top results
    top_snippets = ""
    for i, result in enumerate(search.get("organic_results", [])[:3], 1):
        title = result.get("title", "No title")
        link = result.get("link", "No link")
        snippet = result.get("snippet", "")
        top_snippets += f"{i}. {title}\n{snippet}\n{link}"

    return {"search_results": top_snippets.strip()}


def format_answer(state: State):
    return {
        "final_response": (
            "### Optimized Search Results\n"
            "---------------------------\n"
            f"**Intent:** {state['intent']}\n"
            f"**Optimized Query:** {state['optimized_query']}\n"
            "---------------------------\n"
            f"**Top Results:**\n{state['search_results']}"
        )
    }


# Define langgraph workflow with nodes and edges.
def build_search_workflow():
    # Initializes a langgraph workflow where each node operates on the state.
    wf = StateGraph(State)

    # Add the nodes to the langgraph.
    wf.add_node("extract_intent", extract_intent)
    wf.add_node("optimize_query", optimized_query)
    wf.add_node("perform_search", perform_search)
    wf.add_node("format_answers", format_answer)

    # Add the edges to the langgraph.
    wf.add_edge(START, "extract_intent")
    wf.add_edge("extract_intent", "optimize_query")
    wf.add_edge("optimize_query", "perform_search")
    wf.add_edge("perform_search", "format_answers")
    wf.add_edge("format_answers", END)

    # Compile the graph.
    return wf.compile()


# Run the workflow
if __name__ == "__main__":
    search_qeury = "Best laptops under $1000"

    # The state object is initialized with all empty states except the product_name.
    # These state names must exactly be same as in the State class (see line 32 above).
    state = State(
        query=search_qeury,
        intent="",
        optimized_query="",
        search_results="",
        final_response=""
    )

    # Build and execute the workflow.
    graph_wf = build_search_workflow()
    gr_result = graph_wf.invoke(state)

    # Output results
    print(gr_result["final_response"])
    print("=====================================================")
