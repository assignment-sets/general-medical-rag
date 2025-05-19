from google.genai import types
from google.genai.client import Client
from google import genai


def web_search(client: Client, query: str) -> str:
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
            model="gemini-2.0-flash",
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearchRetrieval())]
            ),
        )
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        # Log the exception or handle it as needed
        return f"[ERROR] An error occurred during web search: {str(e)}"


if __name__ == "__main__":
    client = genai.Client(api_key="<GOOGLE_API_KEY>")
    query = "what are some medications used for Lyme Disease"
    result = web_search(client, query)
    print(result)
