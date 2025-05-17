from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.web_search_manager.web_search_service_gemini import perform_web_search
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# user query
user_query = input(
    "Query : ")


# components
llm1 = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

web_search_runnable = RunnableLambda(perform_web_search)

query_structuring_prompt = ChatPromptTemplate.from_template(
    """Rephrase the user's query into a clear, structured, and concise question suitable for web search, correcting any significant spelling errors response must be one liner.
    User Query: {user_query}
    Structured Query:"""
)

op_parser = StrOutputParser()


# creating the chain
chain = query_structuring_prompt | llm1 | op_parser | web_search_runnable

print(f"parsed response : {chain.invoke({'user_query': user_query})}")
