import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b")
model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b")

# Load your dataset
dataset = load_dataset("csv", data_files="E:/ai/Conversation.csv")

# Define a function to tokenize your dataset
def tokenize(batch):
    return tokenizer(batch['question'], padding='max_length', truncation=True), tokenizer(batch['answer'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='E:/ai/fine_tuned_llama',
    per_device_train_batch_size=2,  # Reduced batch size
    num_train_epochs=3,  # Reduced number of epochs
    fp16=True,  # Keep mixed precision training if CUDA is available
    save_steps=500,
    save_total_limit=2,  # Reduced the number of total checkpoints
    gradient_accumulation_steps=8,  # Added gradient accumulation
)

# Determine if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize a Trainer with your model, training arguments, and tokenized dataset
trainer = Trainer(
    model=model.to(device),  # Ensure the model uses the correct device
    args=training_args,
    train_dataset=tokenized_dataset
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model('E:/ai/trained_model')
