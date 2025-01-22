import os
import time
import json
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from accelerate import init_empty_weights, infer_auto_device_map

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")

# Want to use CUDA (GPU) device if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Testing log file creation # Debugging purpose
try:
    with open("test_log_creation.log", "w") as test_log:
        test_log.write("Log creation test successful.")
    print("Log file creation test passed.")
except Exception as e:
    print(f"Log file creation test failed: {e}")

# Loading the dataset
print("Loading Dataset...")
dataset = load_dataset("json", data_files="test_data.json")

# Loading the tokenizer
model_name = "Salesforce/codegen-2B-multi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Loading the base model with Accelerate Offloading
print("Loading model with Accelerate...")
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_name)

# Generating device_map using accelerate
device_map = infer_auto_device_map(
    model,
    no_split_module_classes=["GPTJBlock"],
    max_memory={0: "10GiB", "cpu": "50GiB"}
)

# Loading model weights with device mapping
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)

# Converting model to empty state with the specified device
model.to_empty(device=device)

# Enabling Gradient Checkpointing
model.gradient_checkpointing_enable()

# Preprocessing the dataset
def preprocess_data(example):
    input_text = f"{example['func']} Label: {example['label']}"
    tokenized = tokenizer(
        input_text,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0],
        "labels": tokenized["input_ids"][0],
    }

tokenized_dataset = dataset.map(preprocess_data, batched=False, remove_columns=dataset["train"].column_names)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# LoRA Configuration
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["attn.qkv_proj", "attn.out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# **Enhanced Training Arguments**
training_args = TrainingArguments(
    output_dir="./codegen_lora_finetuned_final",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=2e-5,
    save_steps=100,
    save_total_limit=2,
    logging_steps=5,
    fp16=True,
    report_to=[],
    weight_decay=0.01,
)

print(f"Weight Decay: {training_args.weight_decay}")
print(f"Batch Size: {training_args.per_device_train_batch_size}")

# Customizing Trainer with Enhanced Logging for insights analysis
class EnhancedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = os.path.join(os.getcwd(), "lora_training_logs.log")
        with open(self.log_file, "w") as log:
            log.write("Step\tLoss\tGPU Memory (MB)\tCPU Usage (%)\tStep Time (s)\n")
        self.gpu_memory_per_step = []

    def log_to_file(self, message):
        """Logs a message to the log file."""
        with open(self.log_file, "a") as log:
            log.write(message + "\n")

    def log_system_metrics(self):
        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        cpu_usage = psutil.cpu_percent()
        return gpu_memory, cpu_usage

    def training_step(self, model, inputs, num_items_in_batch):
        start_time = time.time()
        loss = super().training_step(model, inputs, num_items_in_batch)
        step_time = time.time() - start_time
        gpu_memory, cpu_usage = self.log_system_metrics()

        # Logging metrics to file
        self.gpu_memory_per_step.append(gpu_memory)
        self.log_to_file(f"{self.state.global_step}\t{loss:.4f}\t{gpu_memory:.2f}\t{cpu_usage:.2f}\t{step_time:.4f}")

        return loss

    def train(self, *args, **kwargs):
        total_samples = len(self.train_dataset)
        print(f"Total training samples: {total_samples}")
        self.log_to_file(f"Total Training Samples: {total_samples}")

        unique_samples = set()
        for inputs in self.train_dataset:
            unique_samples.add(tuple(inputs["input_ids"].tolist()))
        self.log_to_file(f"Unique Samples: {len(unique_samples)}")

        total_time = super().train(*args, **kwargs)
        avg_gpu_memory = sum(self.gpu_memory_per_step) / len(self.gpu_memory_per_step)
        self.log_to_file(f"Avg GPU Memory: {avg_gpu_memory:.2f}")
        return total_time

trainer = EnhancedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Training the model
trainer.train()

