import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# ---- Load model + tokenizer ----
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Put in train mode (CPU is fine)
model.train()

# ---- Dummy input ----
text = "Hello, my name is"
inputs = tokenizer(text, return_tensors="pt")

# ---- Forward pass ----
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
print("Forward pass OK. Loss:", loss.item())

# ---- Backward pass (tests autograd) ----
loss.backward()
print("Backward pass OK.")

# ---- One tiny optimizer step ----
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
optimizer.step()
print("Optimizer step OK.")
