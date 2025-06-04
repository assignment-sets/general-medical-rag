from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from modules.utils.Utils import Utils
from modules.app_cache.AppCache import AppCache
from modules.web_search_manager.web_search_service_gemini import web_search
from modules.vector_store_manager.vector_store_service_pinecone import semantic_search


# creating instances of the required models
try:
    embedder = AppCache.get_embedder()
    pinecone_index = AppCache.get_pinecone_index()
    web_search_client = AppCache.get_web_search_model()
    vector_store = AppCache.get_vector_store(
        pinecone_index=pinecone_index, embedder=embedder
    )
    llm_query_classifier = AppCache.get_llm_classifier()
    llm_query_rewriter = AppCache.get_llm_query_rewriter()
    llm_synthesizer = AppCache.get_llm_synthesizer()

except Exception as e:
    print(f"[Initialization Error] {type(e).__name__}: {e}")
    raise SystemExit(1)


def perform_web_search(query: str) -> str:
    def _search():
        print("[Debug] Web search start")
        result = web_search(web_search_client, query)
        print("[Debug] Web search end")
        return result

    return Utils.call_with_timeout(_search, timeout=15, fallback="No results")


def perform_vector_search(query: str) -> str:
    def _search():
        print("[Debug] Vector search start")
        results = semantic_search(vector_store, query)
        print("[Debug] Vector search end")
        return Utils.format_docs(results)

    return Utils.call_with_timeout(_search, timeout=10, fallback="No results")


# --- 0. llm_query_classifier – Bad Query Classifier Chain ---
query_classifier_prompt = ChatPromptTemplate.from_template(
    """
    You are a strict binary classifier.
    Determine if the following query is both:
    1. Clearly related to the medical or healthcare domain, and
    2. Free from any unethical, harmful, or inappropriate content.

    Query: {user_query}

    Respond with a single word: 'true' if both conditions are met, or 'false' otherwise.
    """
)

query_classifier_chain = RunnableParallel(
    is_valid_query=query_classifier_prompt | llm_query_classifier | StrOutputParser()
)

# --- 1. llm_query_rewriter – Query Structuring Chain ---
# Takes the raw user query (as a string) and refines it.
query_structuring_prompt = ChatPromptTemplate.from_template(
    """Given the user's medical query, improve it into a clear, succinct and structured question
    suitable for efficient information retrieval. Focus on key medical terms and symptoms.
        
    User Query: {user_query}
    Structured Query:"""
)

# The llm_query_rewriter_chain takes a dictionary like {"user_query": "text"}
# but we'll usually feed the raw query string directly if it's the first step.
# For combining later, it's often easier if components expect dicts.
# Here, llm_query_rewriter_chain will take the `user_query` string directly.
llm_query_rewriter_chain = (
    query_structuring_prompt | llm_query_rewriter | StrOutputParser()
)


# --- 2. Parallel Retrieval Branches ---
# These branches will operate on the *structured query* output by llm_query_rewriter.

# a. Web Search Branch (RunnableLambda to wrap your custom function)
web_search_runnable = RunnableLambda(perform_web_search)

# b. Vector Search Branch (RunnableLambda to wrap your custom function)
vector_search_runnable = RunnableLambda(perform_vector_search)


# --- 3. Combining Retrieval Outputs ---
# This is implicitly handled by how we feed them into llm_synthesizer.
# We'll construct a dictionary containing all necessary pieces of information.

# --- 4. llm_synthesizer – Final Answer Generation Chain ---
final_answer_prompt_template = """You are a helpful and responsible medical AI assistant talking to an end user patient.

You will be provided with two sources of information by the system:
1. Web Search Context
2. Vector Database Context

Your role is to synthesize or derive accurate, accessible, and medically sound responses for users who are not medical professionals. Therefore if you are able to find technical information or scientific terminologies and detailed analogy of professional in the context do include the knowledge in your response but present in a way that is also understandable to the lay users with proper short explanations.

**IMPORTANT:**
- The context is being provided to help you construct the response not to blindly put them in response without understanding
- Strive for a balance between clarity for lay readers and the clinical depth expected from evidence-based medical references.
- These sources may sometimes contain inaccurate, superstitious, outdated, or misleading information. Use your own reasoning and general medical knowledge to verify the accuracy before including anything in your response.
- If the context appears unreliable or insufficient, you may supplement it with your own internal knowledge — but only if you are confident in its relevance and accuracy.
- If neither the context nor your knowledge provides a reliable basis for a meaningful answer, politely inform the user that you cannot help with this query.
- Always include a gentle reminder to consult a qualified healthcare professional before making any health-related decisions.

---

**Query (used for retrieval):**
{structured_query}

**Retrieved Context for You to Consider:**
--- Web Search Context ---
{web_context}
--- Vector Database Context ---
{vector_context}
---

Using the given context, derive or synthesize a clear, informative, and succinct response to the user's original query.

Final Answer:"""

final_answer_prompt = ChatPromptTemplate.from_template(
    template=final_answer_prompt_template
)

# llm_synthesizer_chain will take a dictionary with keys: structured_query, web_context, vector_context
llm_synthesizer_chain = final_answer_prompt | llm_synthesizer | StrOutputParser()


# --- 5. The Master Chain (Putting it all together) ---

# Uses LCEL for orchestrating the flow.
# The input to the entire chain will be the `user_query` string.

# We need to:
# 1. Get the original query.
# 2. Generate the structured query using llm_query_rewriter (from the original query).
# 3. In parallel, use the structured query to:
#    a. Fetch web results.
#    b. Fetch vector store results.
# 4. Gather structured_query, web_context, and vector_context for llm_synthesizer.

# `itemgetter` is used to pick specific keys from the dictionary to feed into subsequent runnables.

# Step 1: Start with the original query and generate the structured query.
# The input to this part of the chain is the raw user_query string.
# Output will be a dict: {"structured_query": "..."}
chain_with_structured_query = RunnableParallel(
    structured_query=llm_query_rewriter_chain,  # llm_query_rewriter_chain also takes the input user_query string
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
#   "structured_query": "structured query text from llm_query_rewriter",
#   "web_context": "text from web search",
#   "vector_context": "text from vector store"
# }
# This dictionary directly matches the input requirements of `llm_synthesizer_chain`.


# The final RAG chain:
rag_chain = full_context_preparation_chain | llm_synthesizer_chain

if __name__ == "__main__":
    import time

    user_query = "how to treat cancer at home"
    start_time = time.time()
    res = rag_chain.invoke({"user_query": user_query})
    end_time = time.time()
    print(res)
    print(f"[Timing] Total response time: {end_time - start_time:.4f} seconds")
