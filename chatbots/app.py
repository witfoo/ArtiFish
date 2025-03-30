from unsloth import FastLanguageModel
import gradio as gr
import torch
from huggingface_hub import snapshot_download
import huggingface_hub as huggingface_api
from serpapi import GoogleSearch  # Import SERPAPI library
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load environment variables from .env file
from dotenv import load_dotenv
import os
load_dotenv()

# Set the Hugging Face token from the environment variable
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    huggingface_api.login(token=hf_token)
else:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables.")

# Get SERPAPI key from environment variable
serpapi_key = os.getenv("SERPAPI_KEY")
if not serpapi_key:
    raise ValueError("SERPAPI_KEY not found in environment variables.")

# Load the model id from environment variable
model_id = os.getenv("MODEL_ID")
if not model_id:
    raise ValueError("MODEL_ID not found in environment variables.")

# Load the download directory from environment variable
download_dir = os.getenv("DOWNLOAD_DIR")
if not download_dir:
    raise ValueError("DOWNLOAD_DIR not found in environment variables.")

 # Check if the download directory exists, if not create it
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Get the location of the ChromaDB chroma.sqlite3 file from environment variable
chroma_db_path = os.getenv("CHROMA_DB_PATH")
if not chroma_db_path:
    raise ValueError("CHROMA_DB_PATH not found in environment variables.")

# Get the location of the data directory from environment variable
data_dir = os.getenv("DATA_DIR")
if not data_dir:
    raise ValueError("DATA_DIR not found in environment variables.")


print(f"Downloading model {model_id} to {download_dir}")


snapshot_download(repo_id=model_id, local_dir=download_dir)

generation_tokens = 8192 # Max tokens to generate in a response
max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Prepare the model for inference
model = FastLanguageModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input_text"]
    outputs      = examples["output_text"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

def input_tokens(instruction, prompt):
    inputs = tokenizer(
        [
            alpaca_prompt.format(instruction, prompt, " ")
        ], return_tensors="pt").to(model.device)
    return inputs

def serpapi_lookup(query):
    """Perform a SERPAPI lookup for the given query."""
    try:
        search = GoogleSearch({"q": query, "api_key": serpapi_key})
        results = search.get_dict()
        if "organic_results" in results and results["organic_results"]:
            return results["organic_results"][0].get("snippet", "No snippet available.")
        return "No relevant information found."
    except Exception as e:
        return f"Error during SERPAPI lookup: {str(e)}"

def generate_response(instruction, input_text):
    # Perform SERPAPI lookup for the input text
    serpapi_result = serpapi_lookup(input_text)
    
    # Include SERPAPI result in the prompt
    inputs = input_tokens(instruction, f"{input_text}\n\nSERPAPI Result: {serpapi_result}")
    outputs = model.generate(**inputs, max_new_tokens=generation_tokens, use_cache=True)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract the response portion
    response = response.split("### Response:")[1].strip()
    return response

def chatbot(instructions, input_text):
    response = generate_response(instructions, input_text)
    return response

trained_instructions = [
    "Answer this question",
    "Create a JSON artifact from the message",
    "Identify this syslog message",
    "Explain this syslog message",
]

# Update the generate_response function to include vector-based retrieval
def generate_response_with_vectors(instruction, input_text):
    # Perform SERPAPI lookup for the input text
    serpapi_result = serpapi_lookup(input_text)

    # Load the ChromaDB vectorstore
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=HuggingFaceEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant documents

    # Perform vector-based retrieval
    retrieved_docs = retriever.get_relevant_documents(input_text)
    retrieved_texts = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Include SERPAPI result and retrieved texts in the prompt
    combined_input = f"{input_text}\n\nSERPAPI Result: {serpapi_result}\n\nRetrieved Context:\n{retrieved_texts}"
    inputs = input_tokens(instruction, combined_input)
    outputs = model.generate(**inputs, max_new_tokens=generation_tokens, use_cache=True)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract the response portion
    response = response.split("### Response:")[1].strip()
    return response


iface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Dropdown(choices=trained_instructions, label="Instruction"),
        gr.Textbox(lines=2, placeholder="Enter your input here...", label="Input Text")
    ],
    outputs=gr.Textbox(label="Response"),
    title="WitQ Chatbot"
)


app = gr.Blocks()

with app:
    iface.render()


app.launch()