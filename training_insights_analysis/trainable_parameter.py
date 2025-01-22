import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, IA3Config, get_peft_model

# Load the base model
model_name = "Salesforce/codegen-2B-multi"
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Total trainable parameters for the base model
base_trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

# LoRA Configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["attn.qkv_proj", "attn.out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# LoRA Model
lora_model = get_peft_model(base_model, lora_config)
lora_trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

# IA3 Configuration
ia3_config = IA3Config(
    task_type="CAUSAL_LM",
    peft_type="IA3",
    target_modules=["attn.qkv_proj", "attn.out_proj", "mlp.fc_in", "mlp.fc_out"],
    feedforward_modules=["mlp.fc_in", "mlp.fc_out"],
    fan_in_fan_out=False,
    init_ia3_weights=True,
)

# IA3 Model
ia3_model = get_peft_model(base_model, ia3_config)
ia3_trainable_params = sum(p.numel() for p in ia3_model.parameters() if p.requires_grad)

# Percentage reduction in trainable parameters
lora_param_reduction = 100 * (1 - lora_trainable_params / base_trainable_params)
ia3_param_reduction = 100 * (1 - ia3_trainable_params / base_trainable_params)

# Display results
print(f"Base Model Trainable Parameters: {base_trainable_params:,}")
print(f"LoRA Trainable Parameters: {lora_trainable_params:,} ({lora_param_reduction:.2f}% reduction)")
print(f"IA3 Trainable Parameters: {ia3_trainable_params:,} ({ia3_param_reduction:.2f}% reduction)")
