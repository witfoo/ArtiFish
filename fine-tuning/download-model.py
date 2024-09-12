## Download a Huggingface model to local disk.
from huggingface_hub import snapshot_download
from huggingface_hub import login
import huggingface_hub as huggingface_api

# Set the API token and the model id
hugging_face_api_token = ""
download_dir = "/data/models"
model_id = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"

print(f"Downloading model {model_id} to {download_dir}")

# Login to Huggingface using the api token
login(token=hugging_face_api_token)

snapshot_download(repo_id=model_id, local_dir=download_dir)