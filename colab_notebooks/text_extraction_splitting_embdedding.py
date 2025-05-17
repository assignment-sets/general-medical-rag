# notebook url: https://colab.research.google.com/drive/11t02ySnqe3eUOEDudYj_YibegMJ1zU5o#scrollTo=Nzk4cNxFbhHT

# combined code:
import os
import multiprocessing
from typing import List, Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec, Index
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4


def process_single_pdf_for_chunking(args_tuple: Tuple[str, int, int]) -> List[Document]:
    """
    Helper function to load and split a single PDF.
    This function is designed to be called by multiprocessing.Pool.map,
    so it takes a single argument (a tuple in this case).

    Args:
        args_tuple (Tuple[str, int, int]): A tuple containing:
            - pdf_path (str): Path to the PDF file.
            - chunk_size (int): Desired chunk size for text splitting.
            - chunk_overlap (int): Desired overlap for text splitting.
    Returns:
        List[Document]: A list of LangChain Document objects (chunks).
    """
    pdf_path, chunk_size, chunk_overlap = args_tuple

    if not os.path.exists(pdf_path) or not pdf_path.lower().endswith(".pdf"):

        return []

    try:
        loader = PyPDFLoader(pdf_path)

        pages_as_documents = loader.load()

        if not pages_as_documents:

            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        chunks = text_splitter.split_documents(pages_as_documents)

        return chunks
    except Exception as e:
        print(
            f"Worker (PID {os.getpid()}): Error processing PDF {pdf_path}: {e}")
        return []


def load_and_split_pdfs(
    pdf_paths: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_multiprocessing: bool = True
) -> List[Document]:
    """
    Loads document(s) from the given PDF path(s), splits them into
    text chunks using LangChain, and returns them.
    Uses multiprocessing to process multiple PDF files in parallel if enabled.

    Args:
        pdf_paths (List[str]): A list of paths to PDF files.
        chunk_size (int): The maximum size of each text chunk (in characters).
        chunk_overlap (int): The overlap between consecutive chunks (in characters).
        use_multiprocessing (bool): Whether to use multiprocessing for parallel PDF processing.
                                    Defaults to True.

    Returns:
        List[Document]: A list of LangChain Document objects representing text chunks.
    """
    all_chunks: List[Document] = []

    if not pdf_paths:
        print("No PDF paths provided for chunking.")
        return all_chunks

    valid_pdf_paths = []
    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"Warning: PDF file not found at '{path}', skipping.")
            continue
        if not path.lower().endswith(".pdf"):
            print(f"Warning: File '{path}' is not a PDF, skipping.")
            continue
        valid_pdf_paths.append(path)

    if not valid_pdf_paths:
        print("No valid PDF files found to process after filtering.")
        return all_chunks

    process_args_list = [(path, chunk_size, chunk_overlap)
                         for path in valid_pdf_paths]

    if use_multiprocessing and len(valid_pdf_paths) > 0:
        num_files = len(valid_pdf_paths)
        cpu_cores = os.cpu_count()

        num_processes = min(
            num_files, cpu_cores if cpu_cores is not None else 4)

        print(
            f"Starting PDF processing with {num_processes} worker process(es) for {num_files} PDF file(s)...")

        try:
            with multiprocessing.Pool(processes=num_processes) as pool:

                list_of_chunk_lists = pool.map(
                    process_single_pdf_for_chunking, process_args_list)

            for chunk_list_from_one_pdf in list_of_chunk_lists:
                all_chunks.extend(chunk_list_from_one_pdf)
            print(
                f"Multiprocessing finished. Total {len(all_chunks)} chunks from {num_files} PDF(s).")

        except Exception as e:
            print(f"Multiprocessing pool error: {e}")
            print("Falling back to sequential processing...")
            all_chunks = []
            for args_tuple in process_args_list:
                chunks = process_single_pdf_for_chunking(args_tuple)
                all_chunks.extend(chunks)
            print(
                f"Sequential fallback finished. Total {len(all_chunks)} chunks from {num_files} PDF(s).")
    else:
        print(
            f"Starting PDF processing sequentially for {len(valid_pdf_paths)} PDF file(s)...")
        for args_tuple in process_args_list:
            chunks = process_single_pdf_for_chunking(args_tuple)
            all_chunks.extend(chunks)
        print(
            f"Sequential processing finished. Total {len(all_chunks)} chunks from {len(valid_pdf_paths)} PDF(s).")

    return all_chunks


def setup_pinecone_index(
    pinecone_api_key: str,
    index_name: str,
    dimension: int,
    metric: str,
    region: str,
    cloud_provider: str
) -> Optional[Index]:
    try:
        pc = Pinecone(api_key=pinecone_api_key)

        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud_provider,
                    region=region
                )
            )
            print(f"[✅] Created new index: {index_name}")
        else:
            print(
                f"[ℹ️] Index '{index_name}' already exists. Using existing index.")

        return pc.Index(index_name)

    except Exception as e:
        print(f"[❌] Error setting up Pinecone index: {e}")
        return None


def store_embeddings_in_pinecone(
    documents: List[Document],
    pinecone_index: Index,
    embedding_model_name: str
) -> bool:
    try:
        embedder = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vector_store = PineconeVectorStore(
            index=pinecone_index, embedding=embedder)

        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)

        print(f"[✅] Successfully stored {len(documents)} documents.")
        return True

    except Exception as e:
        print(f"[❌] Error storing embeddings in Pinecone: {e}")
        return False


if __name__ == '__main__':

    pdf_paths = [
        "./american-medical-association-family-medical-guide_updated.pdf",
    ]

    chunks = load_and_split_pdfs(pdf_paths)

    pinecone_index = setup_pinecone_index(
        pinecone_api_key="pcsk_3FPFq7_KcLWYRPj41oTn4R8PQGuo5DMdEeFcJtTRJWCTVxSecdqwGSAghBPkLGCdmuDxvf",
        index_name="medical-resrc-rag",
        dimension=384,
        metric="cosine",
        region="us-east-1",
        cloud_provider="aws",
    )

    if not pinecone_index:
        print("[🚫] Index creation failed. Exiting.")
    else:
        store_embeddings_in_pinecone(
            documents=chunks,
            pinecone_index=pinecone_index,
            embedding_model_name="all-MiniLM-L6-v2"
        )
