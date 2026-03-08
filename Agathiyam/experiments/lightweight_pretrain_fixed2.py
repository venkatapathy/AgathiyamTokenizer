import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from huggingface_hub import login

# Optional: login if you have HF token
# login("YOUR_TOKEN_HERE")

MODEL_ID = "google/gemma-3-270m"
LOCAL_TOKENIZER_DIR = "./tokenizer_vocab_aug"  # or None

def load_tokenizer(model_id, local_dir):
    if local_dir and os.path.isdir(local_dir):
        try:
            print(f"Loading tokenizer locally from {local_dir}")
            return AutoTokenizer.from_pretrained(local_dir, use_fast=True, local_files_only=True)
        except Exception as le:
            print("Local load failed:", le)
    try:
        print(f"Loading tokenizer from HF: {model_id}")
        return AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except Exception as he:
        print("HF load failed:", he)
        print("Trying fallback: use_fast=False")
        return AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)

tokenizer = load_tokenizer(MODEL_ID, LOCAL_TOKENIZER_DIR)

print("Tokenizer loaded. Vocab size:", len(tokenizer))

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True
)

# If vocabulary augmentation
if LOCAL_TOKENIZER_DIR and os.path.isdir(LOCAL_TOKENIZER_DIR):
    with open("agathyam_tokens.txt", "r", encoding="utf-8") as f:
        new_tokens = [t.strip() for t in f if t.strip()]
    added = tokenizer.add_tokens(new_tokens)
    print(f"Added {added} new tokens")
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        for tok in new_tokens[:added]:
            try:
                sub = tokenizer.tokenize(tok)
                ids = tokenizer.convert_tokens_to_ids(sub)
                if len(ids) > 0:
                    vec = emb[ids].mean(dim=0)
                    emb[tokenizer.convert_tokens_to_ids(tok)] = vec
            except Exception:
                pass

# Prepare small dataset
train_file = "data/samantar_ta_small.txt"
if not os.path.exists(train_file):
    raise ValueError(f"Training file not found: {train_file}")

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=64
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./results_gemma_regimeB2",
    overwrite_output_dir=True,
    num_train_epochs=1,
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
    train_dataset=train_dataset
)

print("Starting training...")
trainer.train()

print("Saving model & tokenizer...")
trainer.save_model("./gemma_regimeB_model2")
tokenizer.save_pretrained("./gemma_regimeB_tokenizer2")

print("✅ Done.")
