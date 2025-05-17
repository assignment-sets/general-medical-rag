from langchain_core.runnables import RunnableParallel, RunnablePassthrough, itemgetter, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI # Assuming you use this

# --- 0. Prerequisites (Illustrative - you'd define these) ---

# Initialize LLMs (Gemini instances)
# Make sure to set your GOOGLE_API_KEY environment variable
llm1 = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True) # For structuring user query
llm2 = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, convert_system_message_to_human=True) # For final answer

# Define your Pinecone vector store and retriever
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings # Or your preferred embeddings
# embeddings = OpenAIEmbeddings() # Or GeminiEmbeddings if/when available and suitable
# vector_store = PineconeVectorStore(index_name="your-index-name", embedding=embeddings)
# pinecone_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Placeholder for your Google Gemini Web Search API call wrapped as a Runnable
# This function would take a query string and return a string of formatted search results
def perform_web_search(query: str) -> str:
    """
    Placeholder for actual Google Gemini Web Search API call.
    Should fetch, parse, and clean results, returning a single string.
    """
    print(f"[Debug] Web searching for: {query}")
    # In a real scenario:
    # response = google_gemini_web_search_client.search(query)
    # cleaned_results = parse_and_clean(response)
    # return cleaned_results
    return f"Web search results for '{query}':\n- Snippet 1 about {query}\n- Snippet 2 related to {query}"

# Placeholder for formatting retrieved documents (from Pinecone)
def format_docs(docs: list) -> str:
    """
    Placeholder for formatting a list of LangChain Documents into a single string.
    """
    print(f"[Debug] Formatting {len(docs)} vector documents.")
    return "\n\n".join([doc.page_content for doc in docs])

# --- 1. LLM1 – Query Structuring Chain ---
# Takes the raw user query (as a string) and refines it.
query_structuring_prompt = ChatPromptTemplate.from_template(
    """Given the user's medical query, rephrase it into a clear, structured question
    suitable for efficient information retrieval. Focus on key medical terms and symptoms.
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
# Assumes perform_web_search takes the structured query string as input
web_search_runnable = RunnableLambda(perform_web_search)

# b. Vector Search Branch
# Assumes pinecone_retriever takes the structured query string as input
# and format_docs processes the list of Document objects from the retriever
# For demonstration, let's create a dummy retriever
class DummyRetriever:
    def invoke(self, query_string: str):
        print(f"[Debug] Dummy Vector searching for: {query_string}")
        
        return [
            Document(page_content=f"Vector DB doc 1 about {query_string}: Details about condition A."),
            Document(page_content=f"Vector DB doc 2 for {query_string}: Treatment options for B.")
        ]
    async def ainvoke(self, query_string: str): # async version for completeness
        return self.invoke(query_string)

pinecone_retriever_dummy = DummyRetriever() # Replace with your actual pinecone_retriever
vector_search_runnable = pinecone_retriever_dummy | RunnableLambda(format_docs)


# --- 3. Combining Retrieval Outputs ---
# This is implicitly handled by how we feed them into LLM2.
# We'll construct a dictionary containing all necessary pieces of information.

# --- 4. LLM2 – Final Answer Generation Chain ---
final_answer_prompt_template = """You are a helpful medical AI assistant.
Your goal is to synthesize information from various sources to answer the user's query.
Do NOT provide medical advice, diagnosis, or treatment recommendations.
Instead, provide general information based on the retrieved context and suggest consulting a healthcare professional.

Original User Query: {original_query}
Refined Query (used for retrieval): {structured_query}

Combined Retrieved Context:
--- Web Search Context ---
{web_context}
--- Vector Database Context ---
{vector_context}
---

Based *only* on the information above, synthesize a comprehensive, informative, and neutral response to the Original User Query.
If the provided context is insufficient or conflicting, clearly state that.
Remember to advise the user to consult with a qualified healthcare professional for any medical concerns.

Final Answer:"""

final_answer_prompt = ChatPromptTemplate.from_template(final_answer_prompt_template)

# llm2_chain will take a dictionary with keys: original_query, structured_query, web_context, vector_context
llm2_chain = final_answer_prompt | llm2 | StrOutputParser()


# --- 5. The Master Chain (Putting it all together) ---

# This is where the LCEL magic comes in for orchestrating the flow.
# The input to the entire chain will be the `user_query` string.

# We need to:
# 1. Get the original query.
# 2. Generate the structured query using LLM1 (from the original query).
# 3. In parallel, use the structured query to:
#    a. Fetch web results.
#    b. Fetch vector store results.
# 4. Gather original_query, structured_query, web_context, and vector_context for LLM2.

# `RunnablePassthrough.assign` is great for adding new keys to the running dictionary.
# `itemgetter` is used to pick specific keys from the dictionary to feed into subsequent runnables.

# Step 1: Start with the original query and generate the structured query.
# The input to this part of the chain is the raw user_query string.
# Output will be a dict: {"original_query": "...", "structured_query": "..."}
chain_with_structured_query = RunnableParallel(
    original_query=RunnablePassthrough(),  # Passes the input user_query string through
    structured_query=llm1_chain            # llm1_chain also takes the input user_query string
)

# Step 2: Now, use the "structured_query" from the output of `chain_with_structured_query`
# to perform parallel retrievals, while also passing through the existing keys.
# The input to `assign` here is the dict from `chain_with_structured_query`.
# `itemgetter("structured_query")` plucks the structured query string and feeds it to the retrieval runnables.
full_context_preparation_chain = chain_with_structured_query | RunnablePassthrough.assign(
    web_context=itemgetter("structured_query") | web_search_runnable,
    vector_context=itemgetter("structured_query") | vector_search_runnable
)
# The output of `full_context_preparation_chain` will be a dictionary:
# {
#   "original_query": "raw user query text",
#   "structured_query": "structured query text from LLM1",
#   "web_context": "text from web search",
#   "vector_context": "text from vector store"
# }
# This dictionary directly matches the input requirements of `llm2_chain`.


# The final RAG chain:
rag_chain = full_context_preparation_chain | llm2_chain

# --- How to run it (example) ---
if __name__ == "__main__":
    # Example user query
    sample_user_query = "I've been feeling tired lately and have a persistent cough, what could it be?"

    print(f"--- Invoking RAG Chain for query: '{sample_user_query}' ---")

    # The input to the rag_chain is the raw user query string,
    # because `RunnablePassthrough()` in `chain_with_structured_query` expects it,
    # and `llm1_chain` also expects the raw query string via its `{user_query}` prompt variable.
    final_response = rag_chain.invoke(sample_user_query)

    print("\n--- Final Generated Response ---")
    print(final_response)

    # To see the intermediate steps (very useful for debugging):
    print("\n--- Intermediate steps from full_context_preparation_chain ---")
    intermediate_data = full_context_preparation_chain.invoke(sample_user_query)
    import json
    print(json.dumps(intermediate_data, indent=2))