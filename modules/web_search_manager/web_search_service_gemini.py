from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

# Create a client instance for interacting with the Google GenAI API
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def web_search(query: str) -> str:
    """
    This function performs a web search using the Google GenAI API and returns
    the text of the search result.

    Args:
    query (str): The search query to be performed.

    Returns:
    str: The search result text from Google GenAI, or an error message.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    google_search=types.GoogleSearchRetrieval()
                )]
            )
        )
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        # Log the exception or handle it as needed
        return f"An error occurred during web search: {str(e)}"


if __name__ == '__main__':
    # Example usage if you run the module directly
    query = "what are some medications used for Lyme Disease"
    result = web_search(query)
    print(result)
