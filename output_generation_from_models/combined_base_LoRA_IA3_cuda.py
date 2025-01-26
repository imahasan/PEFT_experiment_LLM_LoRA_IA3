"""
This script will generate output from all the models.
Place it in the same root folder where the fine-tuned models are.
Then run the script.

The "max_length" variable was kept in 512 normally.
if it is found that any model is generating partial C or C++ code
this value was increased to see whether a full response is possible or not.
It was increased maximum upto 1024. However, increasing this value increase the inference time also!
"""

import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import transformers

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# Setting device for faster code execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== Base Model =====
def generate_base_model_response(prompt, model_name="Salesforce/CodeGen-2B-multi"):
    print("\nGenerating response with Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=512,
        num_return_sequences=1,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# ===== LoRA Model =====
# modify path the the fine-tuned LoRA folder
def generate_lora_model_response(prompt, adapter_path="./codegen_lora_finetuned"): 
    print("\nGenerating response with LoRA Model...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-multi").to(device)
    peft_model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    peft_model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    generation_config = GenerationConfig(
        max_new_tokens=512,
        num_return_sequences=1,
        do_sample=False,
    )
    with torch.no_grad():
        outputs = peft_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=generation_config,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# ===== IA3 Model =====
# modify path the the fine-tuned IA3 folder
def generate_ia3_model_response(prompt, adapter_path="./codegen_ia3_finetuned"):
    print("\nGenerating response with IA3 Model...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-multi").to(device)
    peft_model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    peft_model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = peft_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Prompt to be used for all models. Chaneg it accordingly.
prompt = "In C or C++, write a program to take two integers as input and find their least common multiple (LCM)."

# ===== Main Execution =====
if __name__ == "__main__":
    # Base Model Response
    base_response = generate_base_model_response(prompt)
    print(f"\nBase Model Response:\n{base_response}")
    print('\n' + '*-*' * 50 + '\n')

    # LoRA Model Response
    lora_response = generate_lora_model_response(prompt)
    print(f"\nLoRA Model Response:\n{lora_response}")
    print('\n' + '*-*' * 50 + '\n')

    # IA3 Model Response
    ia3_response = generate_ia3_model_response(prompt)
    print(f"\nIA3 Model Response:\n{ia3_response}")
    print('\n' + '*-*' * 50 + '\n')
