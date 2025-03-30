# filepath: b:\GitHub\ArtiFish\chatbots\create_database.py
# Update deprecated imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm  # Import tqdm for progress bar
import os
import yaml
import hashlib

# Load settings from settings.yaml
with open("settings.yaml", "r") as file:
    settings = yaml.safe_load(file)
# Get the location of the ChromaDB chroma.sqlite3 file from .env file
chroma_db_path = settings.get("CHROMA_DB_PATH")
if not chroma_db_path:
    raise ValueError("CHROMA_DB_PATH not found in settings.yaml.")

data_dir = settings.get("DATA_DIR")
if not data_dir:
    raise ValueError("DATA_DIR not found in settings.yaml.")


# Function to generate a unique ID for a document
def generate_document_id(doc):
    """Generate a unique ID for a document based on its content and metadata."""
    content = doc.page_content
    source = doc.metadata.get('source', '')
    unique_string = f"{content}_{source}"
    return hashlib.md5(unique_string.encode()).hexdigest()


# Function to scan the data directory and store vectors in ChromaDB
def scan_and_store_vectors(data_dir, chroma_db_path):
    """Recursively scan the data directory, process files, and store vectors in ChromaDB."""
    # Load documents from the data directory
    print(f"Scanning documents in {data_dir}...")
    loader = DirectoryLoader(data_dir, recursive=True, show_progress=True)
    documents = loader.load()

    # Split documents into smaller chunks
    print("Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Generate embeddings using HuggingFace
    embeddings = HuggingFaceEmbeddings()

    # Store vectors in ChromaDB
    print("Storing vectors in ChromaDB...")
    
    # Create Chroma instance with persist_directory
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
    
    # Get existing document IDs
    existing_ids = set(vectorstore.get()["ids"])
    
    # Process documents and add only new ones
    new_docs = []
    new_ids = []
    
    for doc in tqdm(docs, desc="Processing documents"):
        doc_id = generate_document_id(doc)
        if doc_id not in existing_ids:
            new_docs.append(doc)
            new_ids.append(doc_id)
    
    # Add new documents with their IDs
    if new_docs:
        print(f"Adding {len(new_docs)} new documents to ChromaDB...")
        vectorstore.add_documents(new_docs, ids=new_ids)
        print(f"Successfully added {len(new_docs)} new document chunks to ChromaDB at {chroma_db_path}")
    else:
        print("No new documents to add. Database is up to date.")
    
    print(f"Total documents in database: {len(vectorstore.get()['ids'])}")
    return vectorstore

# Call the scan_and_store_vectors function during initialization
vectorstore = scan_and_store_vectors(data_dir, chroma_db_path)
