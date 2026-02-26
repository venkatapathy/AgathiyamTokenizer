import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from datasets import load_dataset
import os

MODEL_ID = "google/gemma-3-270m"  # correct HF model ID

# If you have a local tokenizer folder for your vocab_aug (Regime B), set this:
LOCAL_TOKENIZER_DIR = "./tokenizer_vocab_aug"  # or None

# Load tokenizer
if LOCAL_TOKENIZER_DIR and os.path.isdir(LOCAL_TOKENIZER_DIR):
    print(f"Loading tokenizer from local folder: {LOCAL_TOKENIZER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_TOKENIZER_DIR)
else:
    print(f"Loading tokenizer from Hugging Face: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# Load model, on CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map={"": "cpu"},
    attn_implementation="eager"  # Use eager attention implementation
)

# If using vocab augmentation, then
if LOCAL_TOKENIZER_DIR and os.path.isdir(LOCAL_TOKENIZER_DIR):
    # Load new tokens list
    with open("agathyam_tokens.txt", "r", encoding="utf-8") as f:
        new_tokens = [line.strip() for line in f if line.strip()]
    added = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    # Optionally init embeddings for new tokens
    # Simple neighbor mean or copy average of existing pieces
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        for tok in new_tokens:
            sub = tokenizer.tokenize(tok)
            ids = tokenizer.convert_tokens_to_ids(sub)
            if len(ids) > 0:
                vec = emb[ids].mean(dim=0)
                emb[tokenizer.convert_tokens_to_ids(tok)] = vec

# Prepare dataset
train_file = "data/samantar_ta_small.txt"  # make a small slice for test
assert os.path.exists(train_file), f"{train_file} not found"

# Load the dataset using the 🤗 Datasets library
dataset = load_dataset("text", data_files={"train": train_file})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Fix DataLoader (remove pin_memory argument)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./results_gemma_regimeB2",
    overwrite_output_dir=True,
    num_train_epochs=5,  # Increase epochs
    per_device_train_batch_size=1,
    save_steps=50,
    save_total_limit=1,
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"]
)

trainer.train()

trainer.save_model("./gemma270m_regimeB_model")
tokenizer.save_pretrained("./gemma270m_regimeB_tokenizer")

print("✅ Done.")
