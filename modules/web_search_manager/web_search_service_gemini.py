from langchain_google_genai import ChatGoogleGenerativeAI
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool


def web_search(llm: ChatGoogleGenerativeAI, query: str) -> str:
    """
    Performs a web search using a provided ChatGoogleGenerativeAI LLM instance.

    Args:
    llm (ChatGoogleGenerativeAI): An initialized LLM instance with the desired model.
    query (str): The search query to be performed.

    Returns:
    str: The search result text from Gemini via LangChain, or an error message.
    """
    try:
        response = llm.invoke(
            query,
            tools=[GenAITool(google_search={})],
        )
        return response.content
    except Exception as e:
        return f"[ERROR] An error occurred during web search: {str(e)}"


if __name__ == "__main__":
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key="<GOOGLE_API_KEY>",
    )
    query = "what are some medications used for Lyme Disease"
    result = web_search(llm, query)
    print(result)
