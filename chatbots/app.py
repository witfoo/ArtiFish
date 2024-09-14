from unsloth import FastLanguageModel
import gradio as gr
import torch
from huggingface_hub import snapshot_download
import huggingface_hub as huggingface_api

download_dir = "/model/witq"
model_id = "witfoo/witq-1.0"

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

def generate_response(instruction, input_text):
    inputs = input_tokens(instruction, input_text)
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