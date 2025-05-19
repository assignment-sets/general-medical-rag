import os
from typing import List, Optional
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec, Index
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
from dotenv import load_dotenv
from modules.utils import Utils


load_dotenv()


def setup_pinecone_index(
    pinecone_api_key: str,
    index_name: str,
    dimension: int,
    metric: str,
    region: str,
    cloud_provider: str,
) -> Optional[Index]:
    try:
        pc = Pinecone(api_key=pinecone_api_key)

        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud_provider, region=region),
            )
            print(f"[✅] Created new index: {index_name}")
        else:
            print(f"[ℹ️] Index '{index_name}' already exists. Using existing index.")

        return pc.Index(index_name)

    except Exception as e:
        print(f"[❌] Error setting up Pinecone index: {e}")
        return None


def get_or_create_vector_store(
    pinecone_index: Index, embedder: HuggingFaceEmbeddings
) -> Optional[PineconeVectorStore]:
    try:
        vector_store = PineconeVectorStore(index=pinecone_index, embedding=embedder)
        print("[✅] Successfully fetched vector store")
        return vector_store
    except Exception as e:
        print(f"[❌] Error fetching vector store: {e}")
        return None


def store_embeddings_in_pinecone(
    documents: List[Document], vector_store: PineconeVectorStore
) -> Optional[PineconeVectorStore]:
    try:
        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)

        print(f"[✅] Successfully stored {len(documents)} documents.")
        return vector_store

    except Exception as e:
        print(f"[❌] Error storing embeddings in Pinecone: {e}")
        return None


def semantic_search(vector_store: PineconeVectorStore, query: str) -> List[Document]:
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )
        return retriever.invoke(query)
    except Exception as e:
        print(f"[❌] Error during semantic search: {e}")
        return []


if __name__ == "__main__":
    index = setup_pinecone_index(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        index_name="medical-resrc-rag",
        dimension=384,
        metric="cosine",
        region="us-east-1",
        cloud_provider="aws",
    )

    if not index:
        exit(1)

    embedder = Utils.get_embedder()
    if not embedder:
        print("❌ Failed to get embedder.")
        exit(1)

    vector_store = get_or_create_vector_store(index, embedder)
    if not vector_store:
        exit(1)

    # we already done storing so we skipping
    # store_embeddings_in_pinecone(...)

    results = semantic_search(vector_store, "tell me about Diphtheria")

    if results:
        print("\n✅ Semantic Search Results:\n")
        for i, doc in enumerate(results, start=1):
            print(f"🔹 Result {i}")
            print(f"📄 Source: {doc.metadata.get('source', 'Unknown')}")
            print(
                f"📄 Page: {int(doc.metadata.get('page', 0))} / {int(doc.metadata.get('total_pages', 0))}"
            )
            print(f"🔖 Page Label: {doc.metadata.get('page_label', 'N/A')}")
            print("📝 Content Preview:")
            print(doc.page_content.strip()[:500] + "...")
            print("-" * 80)
    else:
        print("❌ No results found.")

    # else:
    #     documents = [
    #         Document(
    #             page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    #             metadata={"source": "tweet"},
    #         ),
    #         Document(
    #             page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    #             metadata={"source": "news"},
    #         ),
    #     ]

    #     vector_store = store_embeddings_in_pinecone(
    #         documents=documents,
    #         pinecone_index=index,
    #         embedding_model_name="all-MiniLM-L6-v2"
    #     )
