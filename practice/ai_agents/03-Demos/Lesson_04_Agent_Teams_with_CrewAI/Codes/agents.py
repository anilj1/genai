from crewai import Agent
from textwrap import dedent
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from tools.search_tools import SearchTools
from tools.calculator_tools import CalculatorTools


class TravelAgents:
    def __init__(self):
            """
            Initializes the TravelAgents class by setting up two OpenAI GPT models:
            - GPT-3.5 model (`gpt-3.5-turbo`) for generating text-based responses.
            - Both models are initialized with a temperature of 0.7 to control randomness 
              in responses.
            
            The models will be used by the different agents (travel planner, 
            city selection expert, and local tour guide) for generating responses based 
            on their specific goals and tasks.
            """
            #self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
            #self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
            self.OpenAIGPT35 = AzureChatOpenAI(
                azure_endpoint="https://openai-api-management-gw.azure-api.net/deployments/gpt-4o-mini/chat/completions?api-version=2023-12-01-preview",
                api_version="2023-12-01-preview",
                api_key="2ABecnfxzhRg4M5D6pBKiqxXVhmGB2WvQ0aYKkbTCPsj0JLKsZPfJQQJ99BDAC77bzfXJ3w3AAABACOGi3sC",
                azure_deployment="gpt-4o-mini",
                temperature=0.7
            )

            self.OpenAIGPT4 = AzureChatOpenAI(
                azure_endpoint="https://openai-api-management-gw.azure-api.net/deployments/gpt-4o-mini/chat/completions?api-version=2023-12-01-preview",
                api_version="2023-12-01-preview",
                api_key="2ABecnfxzhRg4M5D6pBKiqxXVhmGB2WvQ0aYKkbTCPsj0JLKsZPfJQQJ99BDAC77bzfXJ3w3AAABACOGi3sC",
                azure_deployment="gpt-4o-mini",
                temperature=0.7
            )

            
            
            """self.OpenAIGPT35 = AzureOpenAI(
            api_key="FRqCOtltuLqTUrhGsmk9sVwys2w8Wi7tvA8xx7gn6N9MbAVODExKJQQJ99BDAC77bzfXJ3w3AAABACOGOI0V",
            azure_endpoint="https://sg-testing-pub.openai.azure.com",
            api_version="2023-12-01-preview",
            azure_deployment="gpt-4o-mini",
            temperature=0.7
        )"""


            """self.OpenAIGPT35 = AzureOpenAI(
            openai_api_base="https://sg-testing-pub.openai.azure.com",  # Replace with your full Azure endpoint
            openai_api_version="2023-12-01-preview",  # Or whatever version you're using
            openai_api_key="FRqCOtltuLqTUrhGsmk9sVwys2w8Wi7tvA8xx7gn6N9MbAVODExKJQQJ99BDAC77bzfXJ3w3AAABACOGOI0V",
            openai_api_type="azure",
            deployment_name="gpt-4o-mini"
        )"""
 
            """self.OpenAIGPT4 = AzureOpenAI(
            api_key="FRqCOtltuLqTUrhGsmk9sVwys2w8Wi7tvA8xx7gn6N9MbAVODExKJQQJ99BDAC77bzfXJ3w3AAABACOGOI0V",
            azure_endpoint="https://sg-testing-pub.openai.azure.com",
            api_version="2023-12-01-preview",
            azure_deployment="gpt-4o-mini",
            temperature=0.7
        )"""
            
            
            """self.OpenAIGPT4 = AzureOpenAI(
                openai_api_base="https://sg-testing-pub.openai.azure.com",
                openai_api_version="2023-12-01-preview",
                openai_api_key="FRqCOtltuLqTUrhGsmk9sVwys2w8Wi7tvA8xx7gn6N9MbAVODExKJQQJ99BDAC77bzfXJ3w3AAABACOGOI0V",
                openai_api_type="azure",
                deployment_name="gpt-4o-mini"
            )"""
 



    def expert_travel_agent(self):  
        """
        Sets up the 'Expert Travel Agent' to help plan a 7-day trip.

        This agent can:
        - Build a detailed 7-day itinerary for the trip.
        - Suggest a budget for the entire journey.
        - Offer packing tips.
        - Share important safety advice.

        The agent uses the following tools:
        - `SearchTools.search_internet`: To look up useful travel info online.
        - `CalculatorTools.calculate`: To handle any necessary calculations (like costs).

        The agent works with the GPT-3.5 model (`self.OpenAIGPT35`) to create responses and complete tasks.

        Returns:
            Agent: The fully configured Expert Travel Agent.
        """
        return Agent(
                    role="Expert Travel Agent",
                    backstory= f"""I'm a seasoned expert in travel planning and logistics. With decades of experience, I'm here to help plan your perfect trip.""",
                    goal=f"""Plan a 7-day trip with day-by-day details, including a budget, packing tips, and safety recommendations.""",
                    tools=[
                        SearchTools.search_internet,
                        CalculatorTools.calculate
                        ],
                    verbose=True,
                    llm=self.OpenAIGPT35,
                )


    def city_selection_expert(self):
        """
        Creates a 'City Selection Expert' agent to help choose the best travel destinations.

        This agent can:
        - Analyze travel data to recommend cities based on various factors.
        - Take into account weather, season, prices, and the preferences of the traveler.

        The agent uses the following tool:
        - `SearchTools.search_internet`: To gather city-related information online.

        The agent uses the GPT-3.5 model (`self.OpenAIGPT35`) to generate recommendations and respond to queries.

        Returns:
            Agent: The fully set-up City Selection Expert.
        """
        return Agent(
                    role="City Selection Expert",
                    backstory=dedent(f"""A pro at analyzing travel data to pick the best destinations based on a 
                                            variety of factors.""" ),
                    goal=dedent(f"""
                        Find the top cities to visit, considering weather, seasons, budget, and what the traveler enjoys."""
                    ),
                    tools=[SearchTools.search_internet],
                    verbose=True,
                    llm=self.OpenAIGPT35,
                )
        
    def local_tour_guide(self):
        """
        Sets up a 'Local Tour Guide' agent to provide detailed information about a city.

        This agent can:
        - Share insights about the city's top attractions, customs, and local highlights.
        - Offer the best tips for experiencing the city like a local.

        The agent uses the following tool:
        - `SearchTools.search_internet`: To find local information and details about the city.

        The agent uses the GPT-3.5 model (`self.OpenAIGPT35`) to provide responses and recommendations.

        Returns:
            Agent: The fully set-up Local Tour Guide.
        """
        return Agent(
                role="Local Tour Guide",
                    backstory=dedent(f"""
                                     A knowledgeable local guide who knows everything about the city's attractions, 
                                     culture, and hidden gems."""),
                    goal=dedent(f"""   
                                Give the best insights and recommendations about the selected city for an unforgettable 
                                experience."""),
                    tools=[SearchTools.search_internet],
                    verbose=True,
                    llm=self.OpenAIGPT35,
                )