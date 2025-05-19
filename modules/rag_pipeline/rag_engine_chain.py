import concurrent.futures
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.utils import Utils
from modules.web_search_manager.web_search_service_gemini import web_search
from modules.vector_store_manager.vector_store_service_pinecone import semantic_search



# creating instances of the required models
try:
    embedder = Utils.get_embedder()
    pinecone_index = Utils.get_pinecone_index()
    web_search_client = Utils.get_web_search_model()
    vector_store = Utils.get_vector_store(
        pinecone_index=pinecone_index, embedder=embedder
    )

    llm1 = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
    )

    llm2 = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
    )

except Exception as e:
    print(f"[Startup Error] Failed to initialize models: {e}")
    raise SystemExit(1)


# Helper to add timeout logic to any function
def call_with_timeout(func, *args, timeout: int = 10, fallback: str = "") -> str:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"[⏱️ Timeout] Function '{func.__name__}' exceeded {timeout}s.")
        except Exception as e:
            print(f"[❌ Error] Function '{func.__name__}' raised: {e}")
        return fallback


def perform_web_search(query: str) -> str:
    def _search():
        print(f"[Debug] Web searching for: {query}")
        return web_search(web_search_client, query)

    return call_with_timeout(_search, timeout=10, fallback="No results")


def perform_vector_search(query: str) -> str:
    def _search():
        print(f"[Debug] Vector searching for: {query}")
        results = semantic_search(vector_store, query)
        return Utils.format_docs(results)

    return call_with_timeout(_search, timeout=10, fallback="No results")


# --- 1. LLM1 – Query Structuring Chain ---
# Takes the raw user query (as a string) and refines it.
query_structuring_prompt = ChatPromptTemplate.from_template(
    """Given the user's query, improve it into a clear, succinct and structured question
    suitable for efficient information retrieval. Focus on key medical terms and symptoms.
    
    **important: If the user query is unrelated to the medical or health domain, or involves unethical, harmful, or inappropriate content, please indicate this briefly in the structured query.
    
    User Query: {user_query}
    Structured Query:"""
)

# The llm1_chain takes a dictionary like {"user_query": "text"}
# but we'll usually feed the raw query string directly if it's the first step.
# For combining later, it's often easier if components expect dicts.
# Here, llm1_chain will take the `user_query` string directly.
llm1_chain = query_structuring_prompt | llm1 | StrOutputParser()


# --- 2. Parallel Retrieval Branches ---
# These branches will operate on the *structured query* output by LLM1.

# a. Web Search Branch (RunnableLambda to wrap your custom function)
web_search_runnable = RunnableLambda(perform_web_search)

# b. Vector Search Branch (RunnableLambda to wrap your custom function)
vector_search_runnable = RunnableLambda(perform_vector_search)


# --- 3. Combining Retrieval Outputs ---
# This is implicitly handled by how we feed them into LLM2.
# We'll construct a dictionary containing all necessary pieces of information.

# --- 4. LLM2 – Final Answer Generation Chain ---
final_answer_prompt_template = """You are a helpful medical AI assistant.
Your goal is to synthesize information from various sources to answer the query in your own way.
provide general information based on the retrieved context along with medical advice, diagnosis, or treatment recommendations if available.

**important: suggest consulting a healthcare professional than directly relying on your response as a safety measure when you feel necessary.

**important: if the query is not related to medical domain or knowledge seeking purpose response with a graceful reminder of your purpose and usecase and do not provide any other information out of context as a medical assistant.

**important: if you are unable find any relevant information from any of the provided sources then you are free to fall back to your own knowledge base to generate a response to query.

**important: if you are unable find any relevant information from any of the provided sources and you also do not have any relevant information to provide as a fall back response to the query then please politely let the user know that you can not help regarding given query.

Query (used for retrieval): {structured_query}

Combined Retrieved Context (to be used by you for generating response when possible):
--- Web Search Context ---
{web_context}
--- Vector Database Context ---
{vector_context}
---

Based *only* on the information above, synthesize a comprehensive, informative, and neutral response to the Original User Query.
If the provided context is insufficient or conflicting, clearly state that.
Remember to advise the user to consult with a qualified healthcare professional for any medical concerns.

Final Answer:"""

final_answer_prompt = ChatPromptTemplate.from_template(
    template=final_answer_prompt_template
)

# llm2_chain will take a dictionary with keys: structured_query, web_context, vector_context
llm2_chain = final_answer_prompt | llm2 | StrOutputParser()


# --- 5. The Master Chain (Putting it all together) ---

# Uses LCEL for orchestrating the flow.
# The input to the entire chain will be the `user_query` string.

# We need to:
# 1. Get the original query.
# 2. Generate the structured query using LLM1 (from the original query).
# 3. In parallel, use the structured query to:
#    a. Fetch web results.
#    b. Fetch vector store results.
# 4. Gather structured_query, web_context, and vector_context for LLM2.

# `itemgetter` is used to pick specific keys from the dictionary to feed into subsequent runnables.

# Step 1: Start with the original query and generate the structured query.
# The input to this part of the chain is the raw user_query string.
# Output will be a dict: {"structured_query": "..."}
chain_with_structured_query = RunnableParallel(
    structured_query=llm1_chain,  # llm1_chain also takes the input user_query string
)

# Step 2: Now, use the "structured_query" from the output of `chain_with_structured_query`
# to perform parallel retrievals, while also passing through the existing keys.
# The input to `assign` here is the dict from `chain_with_structured_query`.
# `itemgetter("structured_query")` plucks the structured query string and feeds it to the retrieval runnables.
full_context_preparation_chain = (
    chain_with_structured_query
    | RunnablePassthrough.assign(
        web_context=itemgetter("structured_query") | web_search_runnable,
        vector_context=itemgetter("structured_query") | vector_search_runnable,
    )
)
# The output of `full_context_preparation_chain` will be a dictionary:
# {
#   "structured_query": "structured query text from LLM1",
#   "web_context": "text from web search",
#   "vector_context": "text from vector store"
# }
# This dictionary directly matches the input requirements of `llm2_chain`.


# The final RAG chain:
rag_chain = full_context_preparation_chain | llm2_chain

# --- How to run it (example) ---
if __name__ == "__main__":
    # sample_user_query = "Cure of Measles disease"
    sample_user_query = ""
    # sample_user_query = "who is the best football player ?"

    print(f"--- Invoking RAG Chain for query: '{sample_user_query}' ---")
    final_response = rag_chain.invoke(sample_user_query)

    print("\n--- Final Generated Response ---")
    print(final_response)

    # To see the intermediate steps (very useful for debugging):
    print("\n--- Intermediate steps from full_context_preparation_chain ---")
    intermediate_data = full_context_preparation_chain.invoke(sample_user_query)
    import json

    print(json.dumps(intermediate_data, indent=2))
