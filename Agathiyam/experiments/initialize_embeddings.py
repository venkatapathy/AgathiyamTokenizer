import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-3-270m"   # or meta-llama if you have access
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Load your Tamil tokens
with open("agathyam_tokens.txt", "r", encoding="utf-8") as f:
    new_tokens = [t.strip() for t in f if t.strip()]

# Add them to tokenizer and resize model embeddings
added = tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# Smart embedding initialization: mean of subword embeddings
embed = model.get_input_embeddings().weight.data
for tok in new_tokens:
    pieces = tokenizer.tokenize(tok)
    idxs = tokenizer.convert_tokens_to_ids(pieces)
    if idxs and all(i < embed.shape[0] for i in idxs):
        vec = embed[idxs].mean(dim=0)
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id < embed.shape[0]:
            embed[tok_id] = vec

# Save model and tokenizer if needed
# model.save_pretrained("your_save_path")
# tokenizer.save_pretrained("your_save_path")
