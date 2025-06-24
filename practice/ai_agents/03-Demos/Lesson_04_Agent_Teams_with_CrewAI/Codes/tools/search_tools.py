import json
import requests
from langchain.tools import tool

# Replace this with your actual Serper API key
SERPER_API_KEY = "0e58d080d199ccd7ef09830210c4010a600ca2db"

class SearchTools():

    @tool("Search the internet")
    def search_internet(query):
        """Helps search the internet for a given topic and retrieve relevant results."""
        top_result_to_return = 4
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'content-type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)

        if 'organic' not in response.json():
            return """Apologies, I couldn't locate any results for that query. 
                      The problem might be with your Serper API key."""
        else:
            results = response.json()['organic']
            string = []
            for result in results[:top_result_to_return]:
                try:
                    string.append('\n'.join([
                        f"Title: {result['title']}", f"Link: {result['link']}",
                        f"Snippet: {result['snippet']}", "\n-----------------"
                    ]))
                except KeyError:
                    continue

            return '\n'.join(string)