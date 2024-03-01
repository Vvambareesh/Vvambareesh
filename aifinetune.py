from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the model name
model_name = "openlm-research/open_llama_3b"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load your dataset
dataset = load_dataset("csv", data_files="E:/ai/Conversation.csv")

# Define a function to tokenize your dataset
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True)

# Tokenize your dataset
tokenized_dataset = dataset.map(tokenize, batched=True)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='E:/ai/fine_tuned_llama',
    per_device_train_batch_size=16,
    num_train_epochs=10,
    fp16=True,
    save_steps=500,
    save_total_limit=5
)

# Initialize a Trainer with your model, training arguments, and tokenized dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Check if CUDA is available and if so, use it
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model.to(device)

# Train the model
trainer.train()

# Save the trained model to a specified location (replace 'your_path' with the path you want)
trainer.save_model('E:/ai/trained_model')
