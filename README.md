# ArtiFish
Toolkit for genai in cybersecurity. Details, resources and discussion can be found at https://www.charlesherring.com/ArtiFish

# Installation
- Install CUDA Toolkit 12.4: https://developer.nvidia.com/cuda-12-4-0-download-archive
- Install Unsloth toolkit: https://github.com/unslothai/unsloth
- Install PyTorch https://pytorch.org/get-started/locally
- Install `pip install -r requirements.txt`
- Configure and run the appropriate script

# Scripts
The code in this repo are divided into 3 sub-folders: `dataset-generation`, `fine-tuning` and `chatbots`

## dataset-generation
The files in the `dataset-creation` folder were used to create public facing and private datasts used by WitFoo R&D
- `precinct-to-syslog-dataset.py` - Connects to a WitFoo Precinct Cassandra cluster, analyzes Incidents and Artifacts to create a dataset for training how to translate syslog formats to English.
- `syslog-to-parser-code.py` - Used to take input syslog messages and code examples to create parsers for training a codegen model.
- Public datasets can be found on WitFoo's Huggingface page at https://huggingface.co/witfoo.

## fine-tuning
The files in the `fine-tuning` folder are used to train a model to understand a dataset
- `unsloth-llama3-fine-tune-from-CSV.py` - Fine tuning script that loads a local model and CSV file and writes a newly tuned model.
- `unsloth-llama3-fine-tune-from-dataset.py` - Fine tuning script that loads a local or HF model and HF Dataset and writes a newly tuned model.
- `download-model.py` - Download a Huggingface model to local disk.
- Public fine tuned models that used this script can be found on WitFoo's Huggingface page at https://huggingface.co/witfoo.

## chatbots
The files in the `chatbots` folder create Web User Interfaces to interact with models.
- `unsloth-llama3-instruct-chatbot.py` - Chatbot for interacting with an Unsloth optimized Llama 3 Instruct model
- `huggingface-llama3-instruct-chatbot.py` - Chatbot using HF standard transformers
- `local-model-unsloth-chatbot.py` - Chatbot for interacting with a local, fine-tuned model with Unsloth optimizations. Can run on small GPU (CPU not supported)
- `local-model-transformers-chatbot.py` - Chatbot for interacting with a local, fine-tuned model with standard Transformers. Can run on GPU or CPU.
- `witq-unsloth-chatbot.py` - Chatbot for interacting with WitFoo's Opensource model with Unsloth optimizations. Can run on small GPU (CPU not supported)
- `witq-transformers-chatbot.py` - Chatbot for interacting with WitFoo's Opensource model with standard Transformers. Can run on GPU or CPU.
- Public fine tuned models that used this script can be found on WitFoo's Huggingface page at https://huggingface.co/witfoo.