from typing import List, Optional
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec, Index
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
from dotenv import load_dotenv


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
