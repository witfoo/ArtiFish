# filepath: b:\GitHub\ArtiFish\chatbots\create_database.py
# Update deprecated imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm  # Import tqdm for progress bar
import os

# Load environment variables from .env file
from dotenv import load_dotenv
import os
load_dotenv()

# Get the location of the ChromaDB chroma.sqlite3 file from environment variable
chroma_db_path = os.getenv("CHROMA_DB_PATH")
if not chroma_db_path:
    raise ValueError("CHROMA_DB_PATH not found in environment variables.")

# Get the location of the data directory from environment variable
data_dir = os.getenv("DATA_DIR")
if not data_dir:
    raise ValueError("DATA_DIR not found in environment variables.")

# Function to scan the data directory and store vectors in ChromaDB
def scan_and_store_vectors(data_dir, chroma_db_path):
    """Recursively scan the data directory, process files, and store vectors in ChromaDB."""
    # Load documents from the data directory
    print(f"Scanning documents in {data_dir}...")
    loader = DirectoryLoader(data_dir, recursive=True)
    documents = loader.load()

    # Split documents into smaller chunks
    print("Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Generate embeddings using HuggingFace
    embeddings = HuggingFaceEmbeddings()

    # Store vectors in ChromaDB
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
    print("Storing vectors in ChromaDB...")
    # Add a progress bar for adding documents
    for doc in tqdm(docs, desc="Storing vectors in ChromaDB"):
        vectorstore.add_documents([doc])
    vectorstore.persist()

    return vectorstore

# Call the scan_and_store_vectors function during initialization
vectorstore = scan_and_store_vectors(data_dir, chroma_db_path)
