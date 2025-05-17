import os
import multiprocessing
from typing import List, Optional, Tuple
from pypdf import PdfReader, PdfWriter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class Utils:
    _embedder = None  # class-level cache
    _embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    @staticmethod
    def get_embedder() -> HuggingFaceEmbeddings:
        if Utils._embedder is None:
            print("[ℹ️] Loading embedding model...")
            Utils._embedder = HuggingFaceEmbeddings(
                model_name=Utils._embedding_model_name
            )
            print("[✅] Embedder loaded.")
        return Utils._embedder
    
    @staticmethod
    def remove_leading_and_trailing_pages(
        pdf_path: str,
        pages_to_remove_from_start: int,
        pages_to_remove_from_end: int
    ) -> str | None:
        """
        Removes a specified number of pages from the start and/or end of a PDF file.
        Saves the modified PDF with '_updated' appended to its original name.

        Args:
            pdf_path (str): The full path to the input PDF file.
            pages_to_remove_from_start (int): Number of pages to remove from the beginning.
                                               Must be non-negative.
            pages_to_remove_from_end (int): Number of pages to remove from the end.
                                             Must be non-negative.

        Returns:
            str | None: The path to the newly created PDF if successful,
                        None otherwise (e.g., file not found, invalid page numbers).
        """
        if not os.path.exists(pdf_path):
            print(f"Error: Input PDF file not found at '{pdf_path}'")
            return None

        if not pdf_path.lower().endswith(".pdf"):
            print(
                f"Error: Input file '{pdf_path}' does not appear to be a PDF.")
            return None

        if pages_to_remove_from_start < 0 or pages_to_remove_from_end < 0:
            print("Error: Number of pages to remove cannot be negative.")
            return None

        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)

            if pages_to_remove_from_start + pages_to_remove_from_end > total_pages:
                print(
                    f"Error: Cannot remove {pages_to_remove_from_start} (start) + "
                    f"{pages_to_remove_from_end} (end) = "
                    f"{pages_to_remove_from_start + pages_to_remove_from_end} pages "
                    f"from a PDF with only {total_pages} pages. "
                    "This would result in attempting to remove more pages than exist."
                )
                return None

            writer = PdfWriter()

            # Determine the 0-indexed range of pages to *keep*
            # First page to keep (0-indexed)
            start_keep_index = pages_to_remove_from_start
            # Last page to keep (0-indexed)
            # e.g., total 10 pages (0-9), remove 2 from end.
            # We want to keep up to page (10 - 2 - 1) = index 7.
            end_keep_index = total_pages - pages_to_remove_from_end - 1

            pages_kept_count = 0
            # Add pages to the writer if the range is valid
            if start_keep_index <= end_keep_index:  # Check if there are any pages to keep
                for i in range(start_keep_index, end_keep_index + 1):
                    writer.add_page(reader.pages[i])
                    pages_kept_count += 1

            # If start_keep_index > end_keep_index, it means all pages are effectively removed.
            # In this case, writer.pages will be empty, leading to an empty PDF.

            # Construct the output file path
            base, ext = os.path.splitext(pdf_path)
            output_pdf_path = f"{base}_updated{ext}"

            # Write the output PDF
            with open(output_pdf_path, "wb") as output_file:
                writer.write(output_file)

            if pages_kept_count > 0:
                print(
                    f"Successfully created updated PDF: '{output_pdf_path}' with {pages_kept_count} pages.")
            else:
                # This covers cases where all pages were removed
                print(
                    f"Successfully created an empty PDF (0 pages retained): '{output_pdf_path}'")

            return output_pdf_path

        except Exception as e:
            print(f"An error occurred while processing '{pdf_path}': {e}")
            return None

    @staticmethod
    def _process_single_pdf_for_chunking(args_tuple: Tuple[str, int, int]) -> List[Document]:
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
        pdf_path, chunk_size, chunk_overlap = args_tuple  # Unpack arguments

        # This print can be helpful for debugging multiprocessing
        # print(f"Worker (PID {os.getpid()}): Starting processing for {pdf_path}")

        # Basic file validation (already done in main method, but good for worker too)
        if not os.path.exists(pdf_path) or not pdf_path.lower().endswith(".pdf"):
            # print(f"Worker (PID {os.getpid()}): Invalid or non-existent PDF '{pdf_path}', skipping.")
            return []

        try:
            loader = PyPDFLoader(pdf_path)
            # PyPDFLoader loads each page of the PDF as a separate Document
            pages_as_documents = loader.load()

            if not pages_as_documents:
                # print(f"Worker (PID {os.getpid()}): No content loaded from PDF: {pdf_path}")
                return []

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,      # Use character count for chunk size
                is_separator_regex=False,  # Use default separators like "\n\n", "\n", " ", ""
            )

            # Split the loaded page documents into smaller chunks
            chunks = text_splitter.split_documents(pages_as_documents)

            # print(f"Worker (PID {os.getpid()}): Finished {pdf_path}. Pages: {len(pages_as_documents)}, Chunks: {len(chunks)}")
            return chunks
        except Exception as e:
            print(
                f"Worker (PID {os.getpid()}): Error processing PDF {pdf_path}: {e}")
            return []

    @staticmethod
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

        # Filter out non-existent or non-PDF files before processing
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

        # Prepare arguments for each call to _process_single_pdf_for_chunking
        # Each item in process_args will be a tuple (pdf_path, chunk_size, chunk_overlap)
        process_args_list = [(path, chunk_size, chunk_overlap)
                             for path in valid_pdf_paths]

        if use_multiprocessing and len(valid_pdf_paths) > 0:
            num_files = len(valid_pdf_paths)
            cpu_cores = os.cpu_count()
            # Use at most cpu_cores, but not more processes than files
            num_processes = min(
                num_files, cpu_cores if cpu_cores is not None else 4)

            print(
                f"Starting PDF processing with {num_processes} worker process(es) for {num_files} PDF file(s)...")

            # try-finally for pool is implicitly handled by 'with' statement
            try:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    # pool.map passes each element from process_args_list as a single argument
                    # to Utils._process_single_pdf_for_chunking
                    list_of_chunk_lists = pool.map(
                        Utils._process_single_pdf_for_chunking, process_args_list)

                # list_of_chunk_lists is a list, where each element is the list of chunks from one PDF
                for chunk_list_from_one_pdf in list_of_chunk_lists:
                    all_chunks.extend(chunk_list_from_one_pdf)
                print(
                    f"Multiprocessing finished. Total {len(all_chunks)} chunks from {num_files} PDF(s).")

            except Exception as e:  # Catch potential errors from multiprocessing itself
                print(f"Multiprocessing pool error: {e}")
                print("Falling back to sequential processing...")
                all_chunks = []  # Reset chunks if multiprocessing failed
                for args_tuple in process_args_list:
                    chunks = Utils._process_single_pdf_for_chunking(args_tuple)
                    all_chunks.extend(chunks)
                print(
                    f"Sequential fallback finished. Total {len(all_chunks)} chunks from {num_files} PDF(s).")
        else:
            print(
                f"Starting PDF processing sequentially for {len(valid_pdf_paths)} PDF file(s)...")
            for args_tuple in process_args_list:
                chunks = Utils._process_single_pdf_for_chunking(args_tuple)
                all_chunks.extend(chunks)
            print(
                f"Sequential processing finished. Total {len(all_chunks)} chunks from {len(valid_pdf_paths)} PDF(s).")

        return all_chunks


if __name__ == "__main__":
    pass
