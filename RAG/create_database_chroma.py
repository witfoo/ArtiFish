import os
import re
import yaml
import hashlib
import glob
import logging
import warnings
import sqlite3
from datetime import datetime
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_unstructured import UnstructuredLoader
from langchain.schema import Document
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import PyPDFLoader
import json  # Add this import for JSON processing
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants
SUPPORTED_FILE_TYPES = ['.pdf', '.txt', '.doc', '.epub', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.csv']
HASH_DB_FILENAME = "file_hashes.db"
DOUBLE_CHECK_FILE_PATHS = True  # Set to True to double-check file paths in the vector and hash databases to make sure no files are skipped
Image.MAX_IMAGE_PIXELS = None  # Disable the limit on image size

# Suppress warnings
warnings.filterwarnings("ignore", message="In the future version we will turn default option ignore_ncx to True.")
warnings.filterwarnings("ignore", message="This search incorrectly ignores the root element, and will be fixed in a future version.")
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
logging.getLogger("unstructured").setLevel(logging.ERROR)

# Load settings
with open("settings.yaml", "r") as file:
    settings = yaml.safe_load(file)

# Extract settings
CHROMA_DB_PATH = settings.get("CHROMA_DB_PATH")
DATA_DIR = settings.get("DATA_DIR")

if not CHROMA_DB_PATH:
    raise ValueError("CHROMA_DB_PATH not found in settings.yaml.")
if not DATA_DIR:
    raise ValueError("DATA_DIR not found in settings.yaml.")

DB_ERROR_LOG = os.path.join(CHROMA_DB_PATH, "db_error.log")
HASH_DB_PATH = os.path.join(CHROMA_DB_PATH, HASH_DB_FILENAME)

# Global counters
error_count = 0
processed_files_count = 0
batches_processed = 0


def log_error(message):
    """Log error messages to a file with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DB_ERROR_LOG, "a") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")


def initialize_hash_db():
    """Initialize the SQLite database to store file hashes."""
    conn = sqlite3.connect(HASH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS file_hashes (
            hash TEXT PRIMARY KEY,
            file_path TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def generate_file_hash(file_path):
    """Generate a hash for a file based on its content."""
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except PermissionError as e:
        log_error(f"Permission denied for file {file_path}: {e}")
        return None
vector_paths = []
def load_vector_paths():
    if not DOUBLE_CHECK_FILE_PATHS: return False
    global vector_paths
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=HuggingFaceEmbeddings())
    existing_docs = vectorstore.get(include=["metadatas"])  # Retrieve metadata for all documents
    for metadata in existing_docs["metadatas"]:
        if "source" in metadata:
            if metadata["source"] not in vector_paths:
                vector_paths.append(metadata["source"])
    vectorstore = None

def file_already_processed(file_path):
    global vector_paths
    """Check if a file's hash already exists in the database."""
    file_hash = generate_file_hash(file_path)
    if file_hash is None:
        return True
    conn = sqlite3.connect(HASH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM file_hashes WHERE hash = ?", (file_hash,))
    result = cursor.fetchone()
    conn.close()
    if result is not None:
        if file_path in vector_paths:
            return True        
    return False


def save_file_hash(file_path):
    """Save a file's hash to the database."""
    file_hash = generate_file_hash(file_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(HASH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO file_hashes (hash, file_path, timestamp) VALUES (?, ?, ?)",
                   (file_hash, file_path, timestamp))
    conn.commit()
    conn.close()


def extract_text_from_epub(file_path):
    """Extract text content from an EPUB file."""
    try:
        book = epub.read_epub(file_path)
        text_content = []

        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text_content.append(soup.get_text())

        return "\n".join(text_content)
    except Exception as e:
        log_error(f"Failed to extract text from EPUB file {file_path}: {e}")
        return None

def process_pdf(file_path):
    """Process a PDF file and extract its content."""
    try:
        # Attempt to extract text using PyMuPDF
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        if not text.strip():
            # If no text is found, use OCR as a fallback
            images = convert_from_path(file_path)
            ocr_text = ""
            for image in images:
                ocr_text += pytesseract.image_to_string(image)
            return ocr_text

        return text
    except Exception as e:
        log_error(f"Unexpected error with MuPDF for {file_path}: {e}")

#Load text splitter globally
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def process_file(file_path):
    """Process a file and extract its content."""
    try:
        if file_path.lower().endswith(".pdf"):
            content = process_pdf(file_path)
        elif file_path.lower().endswith(".epub"):
            content = extract_text_from_epub(file_path)
        elif file_path.lower().endswith(".txt"):
            # Try reading the file with UTF-8 encoding, fallback to other encodings if needed
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    log_error(f"Failed to decode file {file_path} with UTF-8 and Latin-1 encodings.")
                    return None
        elif file_path.lower().endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                # Flatten the JSON content into a string
                content = json.dumps(json_data, indent=2)
        else:
            loader = UnstructuredLoader([file_path])
            docs = loader.load()
            content = docs[0].page_content if docs else None

        if content:
            # Split the content into smaller chunks
            chunks = text_splitter.split_text(content)
            return chunks
        return None
    except Exception as e:
        log_error(f"Error processing file {file_path}: {e}")
        return None


def process_batch(batch, vectorstore, existing_ids, progress_bars):
    """Process a batch of files and store their vectors."""
    global error_count, processed_files_count, batches_processed
    processed_documents = []

    for file_path in batch:
        try:
            # Update the progress bar description with the current file being processed
            progress_bars["total"].set_description(f"Processing: {os.path.basename(file_path)}")
            content = process_file(file_path)
            if content:
                for chunk in content:
                    doc = Document(page_content=chunk, metadata={"source": file_path})
                    doc_id = hashlib.md5(chunk.encode()).hexdigest()
                    if doc_id not in existing_ids:
                        processed_documents.append(doc)
                        existing_ids.add(doc_id)
            else:
                log_error(f"Failed to process file {file_path}: No content extracted.")
                error_count += 1
                progress_bars["error"].update(1)
        except Exception as e:
            log_error(f"Error processing file {file_path}: {e}")
            error_count += 1
            progress_bars["error"].update(1)

        save_file_hash(file_path)
        processed_files_count += 1
        progress_bars["total"].update(1)
        # Refresh all progress bars
        progress_bars["total"].refresh()
        progress_bars["error"].refresh()
        progress_bars["batches"].refresh()

    # Add documents to the vectorstore in smaller sub-batches
    max_batch_size = 1000  # Set a safe maximum batch size
    for i in range(0, len(processed_documents), max_batch_size):
        sub_batch = processed_documents[i:i + max_batch_size]
        try:
            vectorstore.add_documents(sub_batch)
        except Exception as e:
            log_error(f"Error adding documents to vectorstore: {e}")
            # Exit the process
            vectorstore = None
            raise e

    batches_processed += 1
    progress_bars["batches"].update(1)


def process_batch_threaded(batch, vectorstore, existing_ids, progress_bars):
    """Wrapper for processing a batch in a threaded environment."""
    process_batch(batch, vectorstore, existing_ids, progress_bars)


def scan_and_store_vectors(data_dir, chroma_db_path):
    global error_count, processed_files_count, batches_processed
    """Scan the data directory, process files, and store vectors in ChromaDB using threading."""
    initialize_hash_db()
    print(f"Opening error log at {DB_ERROR_LOG}")
    print(f"Scanning directory: {data_dir}")
    file_paths = glob.glob(f"{data_dir}/**/*.*", recursive=True)
    print(f"Found {len(file_paths)} files.")
    file_paths = [fp for fp in file_paths if any(fp.lower().endswith(ext) for ext in SUPPORTED_FILE_TYPES)]
    print(f"Filtered to {len(file_paths)} supported files.")
    print(f"Loading vector paths...")
    load_vector_paths()
    print(f"Checking for already processed files...")
    ## If no entries in the database, process all files
    conn = sqlite3.connect(HASH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM file_hashes")
    count = cursor.fetchone()[0]
    conn.close()
    if count == 0:
        print("No entries in the database. Processing all files.")
    else:
        print(f"Database has {count} entries. Filtering out already processed files.")
        # Filter out files that have already been processed
        file_paths = [fp for fp in file_paths if not file_already_processed(fp)]
        print(f"Filtered to {len(file_paths)} unprocessed files.")

    embeddings = HuggingFaceEmbeddings()
    print(f"Initializing ChromaDB at {chroma_db_path}")
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
    existing_ids = set(vectorstore.get()["ids"])

    batch_size = 10
    batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]

    with tqdm(total=len(file_paths), desc="Total Progress", unit="file") as total_bar, \
         tqdm(total=len(file_paths), desc="Errors", unit="error") as error_bar, \
         tqdm(total=len(batches), desc="Batches", unit="batch") as batch_bar, \
         ThreadPoolExecutor(max_workers=2) as executor:
        
        progress_bars = {"total": total_bar, "error": error_bar, "batches": batch_bar}
        futures = {executor.submit(process_batch_threaded, batch, vectorstore, existing_ids, progress_bars): batch for batch in batches}

        for future in as_completed(futures):
            try:
                future.result()  # Wait for the thread to complete
            except Exception as e:
                log_error(f"Unhandled exception in thread: {e}")
                error_count += 1
                error_bar.update(1)
    # Close the vectorstore connection
    print("Closing ChromaDB connection and vacuuming database.")
    vectorstore.persist()
    vectorstore = None

    print(f"Total errors: {error_count}")
    print(f"Total documents in vectorstore: {len(vectorstore.get()['ids'])}")
    return vectorstore


if __name__ == "__main__":
    try:
        scan_and_store_vectors(DATA_DIR, CHROMA_DB_PATH)
    except Exception as e:
        log_error(f"Unhandled exception: {e}")
        print(f"An error occurred: {e}")
