import os
import pandas as pd
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, TextDataset, Trainer, TrainingArguments

# Load your dataset
csv_file_path = "E:/ai/Conversation.csv"
df = pd.read_csv(csv_file_path)

# Concatenate all text from the training data into a single string
train_text = "\n".join(df["question"].tolist() + df["answer"].tolist())

# Save the concatenated text to a file
train_file_path = "E:/ai/training_data.txt"
with open(train_file_path, "w", encoding="utf-8") as file:
    file.write(train_text)

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Create TextDataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file_path,
    block_size=128,
)

# Training arguments
training_args = TrainingArguments(
    output_dir="E:/ai/trained_model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=train_dataset.collate,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()
