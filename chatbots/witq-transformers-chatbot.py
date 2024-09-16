## Chatbot for interacting with WitFoo's Opensource model with standard Transformers. Can run on GPU or CPU.
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr

model_id = "witfoo/witq-1.0"
dtype = torch.float16 # float16 for Tesla T4, V100, bfloat16 for Ampere+ 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",
)
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

preamble = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."



def input_tokens(instruction, prompt):
    messages = [
        {"role": "system", "content": preamble + " " + instruction},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    return inputs



def generate_response(instruction, input_text):
    input_ids = input_tokens(instruction, input_text)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    # Extract the response portion
    response = outputs[0][input_ids.shape[-1]:]
    result = tokenizer.decode(response, skip_special_tokens=True)
    return result

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