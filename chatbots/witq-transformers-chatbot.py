## Chatbot for interacting with WitFoo's Opensource model with standard Transformers. Can run on GPU or CPU.
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import gradio as gr
import os
import torch

## Configuration Settings
user_name = "admin"
password = "changme"
model_id = "witfoo/witq-1.0"
flagged_responses_dir = "/mnt/data/flagged_responses"
generation_tokens = 8192 # Max tokens to generate in a response
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can 

# If the flagged responses directory does not exist, create it
if not os.path.exists(flagged_responses_dir):
    os.makedirs(flagged_responses_dir)

# Clear torch cache
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

#Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer.to(device)


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
    "Identify this syslog message",
    "Write a WitFoo Parser for this product, syslog message",
    "Create a test for this WitFoo Parser",
    "Explain this code"
]

iface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Dropdown(choices=trained_instructions, label="Instruction"),
        gr.Textbox(lines=2, placeholder="Enter your input here...", label="Input Text")
    ],
    outputs=gr.Textbox(label="Response"),
    title="SPD-13 Chatbot",
    allow_flagging="manual",
    flagging_dir=flagged_responses_dir
)


app = gr.Blocks()

with app:
    iface.render()


app.launch(server_name="0.0.0.0", server_port=80, auth=(user_name, password))