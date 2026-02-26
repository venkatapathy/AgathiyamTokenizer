import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from accelerate import init_empty_weights

# 🔹 Use lightweight Gemma-270M
MODEL_ID = "hf-internal-testing/tiny-random-gpt2"  

# Load tokenizer (replace this with your Regime B/C tokenizer if needed)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load model on CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map={"": "cpu"}
)

# Prepare dataset (use your Tamil dataset instead of sample.txt)
train_file = "sample.txt"  # <-- replace with your Tamil corpus file
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128
)

# Masked language modeling data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training args (very small so it runs on CPU)
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=50,
    save_total_limit=1,
    logging_steps=10,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
train_dataset=train_dataset)

# Train
trainer.train()

# Save updated model + tokenizer
trainer.save_model("./gemma270m_regimeB")
tokenizer.save_pretrained("./gemma270m_regimeB")

print("✅ Training complete. Model saved at ./gemma270m_regimeB")
