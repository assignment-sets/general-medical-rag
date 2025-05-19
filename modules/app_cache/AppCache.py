import os
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from google.genai.client import Client
from modules.vector_store_manager.vector_store_service_pinecone import (
    setup_pinecone_index,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Index
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()


class AppCache:
    _embedder = None
    _web_search_client = None
    _pinecone_index = None
    _vector_store = None
    _llm_query_classifier = None
    _llm_query_rewriter = None
    _llm_synthesizer = None

    @staticmethod
    def get_llm_classifier() -> ChatGoogleGenerativeAI:
        if AppCache._llm_query_classifier is None:
            try:
                print("[ℹ️] Loading LLM classifier...")
                AppCache._llm_query_classifier = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-lite",
                    temperature=0.2,
                )
                print("[✅] LLM classifier loaded.")
            except Exception as e:
                print(f"[❌] Failed to load LLM classifier: {e}")
                AppCache._llm_query_classifier = None

        return AppCache._llm_query_classifier

    @staticmethod
    def get_llm_query_rewriter() -> ChatGoogleGenerativeAI:
        if AppCache._llm_query_rewriter is None:
            try:
                print("[ℹ️] Loading LLM rewriter...")
                AppCache._llm_query_rewriter = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-lite",
                    temperature=0.3,
                )
                print("[✅] LLM rewriter loaded.")
            except Exception as e:
                print(f"[❌] Failed to LLM rewriter: {e}")
                AppCache._llm_query_rewriter = None

        return AppCache._llm_query_rewriter

    @staticmethod
    def get_llm_synthesizer() -> ChatGoogleGenerativeAI:
        if AppCache._llm_synthesizer is None:
            try:
                print("[ℹ️] Loading LLM synthesizer...")
                AppCache._llm_synthesizer = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.6,
                )
                print("[✅] LLM synthesizer loaded.")
            except Exception as e:
                print(f"[❌] Failed to LLM synthesizer: {e}")
                AppCache._llm_synthesizer = None

        return AppCache._llm_synthesizer

    @staticmethod
    def get_pinecone_index() -> Optional[Index]:
        if AppCache._pinecone_index is None:
            try:
                print("[ℹ️] Loading pinecone index...")
                AppCache._pinecone_index = setup_pinecone_index(
                    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                    index_name="medical-resrc-rag",
                    dimension=384,
                    metric="cosine",
                    region="us-east-1",
                    cloud_provider="aws",
                )
                print("[✅] pinecone index loaded.")
            except Exception as e:
                print(e)
                AppCache._pinecone_index = None

        return AppCache._pinecone_index

    @staticmethod
    def get_embedder() -> HuggingFaceEmbeddings:
        if AppCache._embedder is None:
            try:
                print("[ℹ️] Loading embedding model...")
                AppCache._embedder = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                print("[✅] Embedder loaded.")
            except Exception as e:
                print(f"[❌] Failed to load embedder: {e}")
                AppCache._embedder = None
        return AppCache._embedder

    @staticmethod
    def get_vector_store(
        pinecone_index: Index, embedder: HuggingFaceEmbeddings
    ) -> Optional[PineconeVectorStore]:
        if AppCache._vector_store is None:
            try:
                print("[ℹ️] Loading vector store...")
                AppCache._vector_store = PineconeVectorStore(
                    index=pinecone_index, embedding=embedder
                )
                print("[✅] Successfully fetched vector store")
            except Exception as e:
                print(f"[❌] Error fetching vector store: {e}")
                AppCache._vector_store = None

        return AppCache._vector_store

    @staticmethod
    def get_web_search_model() -> Client:
        if AppCache._web_search_client is None:
            try:
                print("[ℹ️] Loading web search model...")
                AppCache._web_search_client = genai.Client(
                    api_key=os.getenv("GOOGLE_API_KEY")
                )
                print("[✅] Successfully fetched web search model")
            except Exception as e:
                print(f"[❌] Exception creating web search client: {e}")
                AppCache._web_search_client = None

        return AppCache._web_search_client
