from unsloth import FastLanguageModel
import gradio as gr
import torch
from huggingface_hub import snapshot_download
from serpapi import GoogleSearch  # Import SERPAPI library
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
import yaml
from langchain_community.vectorstores import Cassandra
from cassandra.cluster import Cluster
import cassio

MAX_DOCUMENTS_TO_FIND = 8

debug_enabled = True # Set to True to enable debug messages

# Load settings from settings.yaml
with open("settings.yaml", "r") as file:
    settings = yaml.safe_load(file)
# Get the location of the ChromaDB chroma.sqlite3 file from .env file
CASSANDRA_CONTACT_POINTS = settings.get("CASSANDRA_CONTACT_POINTS")
CASSANDRA_KEYSPACE = settings.get("CASSANDRA_KEYSPACE")
data_dir = settings.get("DATA_DIR")
if not data_dir:
    raise ValueError("DATA_DIR not found in settings.yaml.")

model_id = settings.get("MODEL_ID")
if not model_id:
    raise ValueError("MODEL_ID not found in settings.yaml.")

download_dir = settings.get("DOWNLOAD_DIR")
if not download_dir:
    raise ValueError("DOWNLOAD_DIR not found in settings.yaml.")

serpapi_key = settings.get("SERPAPI_KEY")
if not serpapi_key:
    raise ValueError("SERPAPI_KEY not found in settings.yaml.")

if not CASSANDRA_CONTACT_POINTS:
    raise ValueError("CASSANDRA_CONTACT_POINTS not found in settings.yaml.")
if not CASSANDRA_KEYSPACE:
    raise ValueError("CASSANDRA_KEYSPACE not found in settings.yaml.")

if debug_enabled: print(f"Downloading model {model_id} to {download_dir}")


snapshot_download(repo_id=model_id, local_dir=download_dir)

generation_tokens = 8192 # Max tokens to generate in a response
max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = download_dir,
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
        if debug_enabled: print(f"Performing SERPAPI lookup for: {query}...")
        search = GoogleSearch({"q": query, "api_key": serpapi_key})
        results = search.get_dict()
        if "organic_results" in results and results["organic_results"]:
            # Return the full first result object instead of just the snippet
            return results["organic_results"]
        return {"snippet": "No relevant information found.", "link": "No link available"}
    except Exception as e:
        return {"snippet": f"Error during SERPAPI lookup: {str(e)}", "link": "Error occurred"}

def generate_response(instruction, input_text):
    # Clear the cache to free up memory
    torch.cuda.empty_cache()
    
    # Get both response and sources from vector function
    vector_results, sources = generate_response_with_vectors(instruction, input_text)
    
    # Include SERPAPI result in the prompt
    inputs = input_tokens(instruction, f"{input_text}\n\nSERPAPI Result: {vector_results}")
    outputs = model.generate(**inputs, max_new_tokens=generation_tokens, use_cache=True)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract the response portion
    response = response.split("### Response:")[1].strip()
    # Clear the cache to free up memory
    torch.cuda.empty_cache()
    return response, sources

def chatbot(instructions, input_text):
    response, sources = generate_response(instructions, input_text)
    return response, sources

trained_instructions = [
    "Answer this question",
    "Create a JSON artifact from the message",
    "Identify this syslog message",
    "Explain this syslog message",
]

# Load the ChromaDB vectorstore from a server
embeddings = HuggingFaceEmbeddings()
print("Initializing Cassandra vectorstore...")
cluster = Cluster([CASSANDRA_CONTACT_POINTS],default_retry_policy=0)
session = cluster.connect()
cassio.init(session=session, keyspace=CASSANDRA_KEYSPACE)
vectorstore = Cassandra(embedding=embeddings, session=session, keyspace=CASSANDRA_KEYSPACE, table_name="documents")
# Update the generate_response function to include vector-based retrieval
def generate_response_with_vectors(instruction, input_text):
    global vectorstore
    retrieved_docs_location = []
    if debug_enabled: print(f"Checking Cassandra Vectors...")
    # Start by getting all vectors with similarity scores below 0.5
    docs_with_similarity = vectorstore.similarity_search_with_score(input_text, k=MAX_DOCUMENTS_TO_FIND*5)
    if debug_enabled: print(f"Found {len(docs_with_similarity)} similar vectors in Cassandra.")
    # Filter out documents with similarity scores below 0.9
    similar_docs = []
    for doc, score in docs_with_similarity:
        if score < 0.8:
            if debug_enabled: print(f"Document: {doc.metadata.get('source', 'Unknown')} (skipped) with score: {score}")
            continue
        # Add the document to the similar_docs list if it has a good score
        similar_docs.append(doc)
        if debug_enabled: print(f"Document: {doc.metadata.get('source', 'Unknown')} (passed)")
    if debug_enabled: print(f"Found {len(similar_docs)} similar vectors with good scores in Cassandra.")
    vector_docs = MAX_DOCUMENTS_TO_FIND - 1
    vector_fetch_k = vector_docs * 2
    docs = vectorstore.max_marginal_relevance_search(input_text, k=vector_docs, fetch_k=vector_fetch_k)
    filtered_docs = []
    for doc in docs:
        # If the doc is not in the similar_docs list, skip it
        if doc not in similar_docs:
            if debug_enabled: print(f"Document: {doc.metadata.get('source', 'Unknown')} (skipped)")
            continue
        retrieved_texts = "\n\n".join([doc.page_content for doc in filtered_docs])
        if len(retrieved_texts) >= 7000:
            break
        filename = doc.metadata.get("source", "Unknown")
        retrieved_docs_location.append(filename)
        filtered_docs.append(doc)
        if debug_enabled: print(f"Document: {filename} (passed)")

    retrieved_texts = "\n\n".join([doc.page_content for doc in filtered_docs])


    # Truncate the retrieved texts to fit within the model's sequence length
    max_context_length = 8192  # Adjust based on your model's max_seq_length
    if len(retrieved_texts) > max_context_length:
        retrieved_texts = retrieved_texts[:max_context_length]
        if debug_enabled: print(f"Truncated retrieved texts to {max_context_length} characters.")

    
    used_docs = len(filtered_docs)
    if debug_enabled: print(f"Found {used_docs} relevant vectors in Cassandra.")
    if debug_enabled: print("Using SERPAPI to enhance response.")
    # Perform SERPAPI lookup for the input text
    serpapi_results = serpapi_lookup(input_text)
    if debug_enabled: print(f"SERPAPI lookup returned {len(serpapi_results)} possible results.")
    text_results = []
    total_records = 0
    max_serpapi_results = 3
    for serpapi_result in serpapi_results:
        # Add the SERPAPI reference locations to the retrieved_docs_location list
        url = serpapi_result.get("link", "No link available.")
        retrieved_docs_location.append(url)
        # Add the SERPAPI snippet to the text_results list
        text_results.append(serpapi_result.get("snippet", "No snippet available."))
        total_records += 1
        if total_records >= max_serpapi_results:
            break
    if debug_enabled: print(f"SERPAPI lookup returned {len(text_results)} results.")
    # Join the SERPAPI snippets into a single string
    serpapi_result = "\n\n".join(text_results)
    # Add the SERPAPI result to the retrieved texts
    serp_texts = f"{serpapi_result}\n\n{retrieved_texts}"
    # Use the snippet in the combined input
    combined_input = f"{input_text}\n\nSERPAPI Result: {serp_texts}\n\nRetrieved Context:\n{retrieved_texts}"

    # Include SERPAPI result and retrieved texts in the prompt
    instruction_preamble = "You are a helpful assistant. Use the following context to answer the question. "
    instruction_preamble += "Retrieved Context is to be weighted more than SERPAPI Result. "
    instruction = f"{instruction_preamble}\n\n{instruction}"
    inputs = input_tokens(instruction, combined_input)
    outputs = model.generate(**inputs, max_new_tokens=generation_tokens, use_cache=True)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract the response portion
    response = response.split("### Response:")[1].strip()
    # Deduplicate the retrieved_docs_location list
    retrieved_docs_location = list(set(retrieved_docs_location))
    # Convert the list to a bulleted list with Markdown hyperlinks
    retrieved_docs_location_formatted = []
    for doc in retrieved_docs_location:
        retrieved_docs_location_formatted.append(f"- {doc}")

    # Join the formatted links
    retrieved_docs_location = "\n".join(retrieved_docs_location_formatted)
    return response, retrieved_docs_location



    combined_input = f"{input_text}\n\nRetrieved Context:\n{retrieved_texts}"

    instruction_preamble = "You are a helpful assistant. Use the following context to answer the question. "
    instruction_preamble += "Retrieved Context is to be weighted more than SERPAPI Result. "
    instruction = f"{instruction_preamble}\n\n{instruction}"
    inputs = input_tokens(instruction, combined_input)
    outputs = model.generate(**inputs, max_new_tokens=generation_tokens, use_cache=True)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    response = response.split("### Response:")[1].strip()
    retrieved_docs_location = list(set(retrieved_docs_location))
    retrieved_docs_location_formatted = [f"- {doc}" for doc in retrieved_docs_location]
    retrieved_docs_location = "\n".join(retrieved_docs_location_formatted)
    return response, retrieved_docs_location


iface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Dropdown(choices=trained_instructions, label="Instruction"),
        gr.Textbox(lines=2, placeholder="Enter your input here...", label="Input Text")
    ],
    outputs=[
        gr.Markdown(label="Response"),
        gr.Textbox(label="Sources")
    ],
    title="Chatbot"
)

# Clear CUDA cache before launching the app
torch.cuda.empty_cache()

app = gr.Blocks()

with app:
    iface.render()


app.launch()