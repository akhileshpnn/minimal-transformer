
# **README.md**

# Minimal Transformer (Tiny GPT-Style Model)

This project provides a **minimal implementation** of a GPT-1/GPT-2–style transformer in PyTorch.

---

## 🚀 Features

* Tiny GPT-style transformer implemented from scratch in PyTorch
* Manual components:
  * Token + positional embeddings
  * Multi-Head Self-Attention
  * LayerNorm + MLP blocks
  * Causal masking for autoregressive modeling
* Compatible with Hugging Face tokenizers and datasets
* Lightweight and easy to modify for experimentation

---

## 🧪 Conda Environment Setup

### **Environment name:** `tiny_trans`

```bash
conda env create -f environment.yaml
conda activate tiny_trans
```

### **Environment includes:**

* Python 3.10
* PyTorch 2.1.2 CPU-only
* torchvision 0.16.2, torchaudio 2.1.2
* Transformers 4.35.2
* numpy < 2
* tqdm

> Windows, CPU-only configuration — avoids PyTorch DLL / NumPy 2.x issues.

---

## ▶️ Training / Testing

Run the minimal transformer script:

```bash
~\model\minimal_transformer.py
```

Minimal code execution:

```python
text = "I want to learn transformer models."
input_ids = encode(text)

with torch.no_grad():
	logits = model(input_ids)

pred_ids = torch.argmax(logits, dim=-1)

decoded = decode(pred_ids)

print('input: '+ text)

print('output: '+ decoded)
```

example output:
```
input: I want to learn transformer models.
output: ,kvvakkkckcauolkkeklhzookseokcosj, 
```
The output is random since the model is unstrained.
---


---

## 📚 Purpose

* Understand GPT-style transformer mechanics
* Study attention
* Run controlled experiments on small-scale models