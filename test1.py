import os
import torch
from torch import cuda
from transformers import GPTNeoConfig, GPTNeoForCausalLM, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Specify the paths
model_path = r"D:\model_data\models--EleutherAI--gpt-neo-1.3B"
dataset_path = r"E:\ai\Conversation.csv"
output_model_path = r"E:\ai\trained_model"

# Load the model configuration
config = GPTNeoConfig.from_json_file(os.path.join(model_path, "snapshots", "8282180b53cba30a1575e49de1530019e5931739", "config.json"))

# Try loading the model from an alternative file
model_file_path = os.path.join(model_path, "snapshots", "8282180b53cba30a1575e49de1530019e5931739", "pytorch_model.bin")
if not os.path.exists(model_file_path):
    model_file_path = os.path.join(model_path, "snapshots", "8282180b53cba30a1575e49de1530019e5931739", "model.safetensors")

# Load the model
model = GPTNeoForCausalLM(config=config)
model.load_state_dict(torch.load(model_file_path, map_location="cuda" if cuda.is_available() else "cpu"))

# Move the model to CUDA if available
model.to("cuda" if cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Load the dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=dataset_path,
    block_size=128  # Adjust block_size as needed
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_model_path,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Training
trainer.train()

# Save the trained model
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
