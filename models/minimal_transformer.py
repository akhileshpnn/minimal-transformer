import torch
import torch.nn as nn
import torch.nn.functional as F

# positional encoding (classic sin/cos)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# multi-head self-attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape

        # Compute Q,K,V
        Q = self.Wq(x)  # (B, T, D)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Hd)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B,H,T,T)
        weights = F.softmax(scores, dim=-1)  # (B,H,T,T)
        out = weights @ V  # (B,H,T,Hd)

        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B,T,D)
        return self.Wo(out)


# transformer block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# full minimal transformer
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=256, depth=2, max_len=500):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
        for _ in range(depth)])
        self.ln = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_emb(x)
        x = self.pos_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return self.output(x)  # logits


if __name__ == "__main__":
    
    ## example with sentences as input and outputs
    ## sentence - tokens - model - logits - tokens - sentence
    
    # list of characters in the input and output sentence
    chars = list("abcdefghijklmnopqrstuvwxyz .,")
    
    vocab_size = len(chars)
    # print("vocab size:", vocab_size)
    
    model = MiniTransformer(vocab_size=len(chars), d_model=64, num_heads=4, d_ff=128, depth=2)
    
    stoi = {c:i for i,c in enumerate(chars)}
    itos = {i:c for i,c in enumerate(chars)}
    

    def encode(text):
        ids = [stoi[c] for c in text.lower()]
        return torch.tensor([ids], dtype=torch.long)
    
    def decode(ids):
        ids = ids.squeeze().tolist()
        return "".join(itos[i] for i in ids)
    
    #%%
    text = "I want to learn transformer models."
    input_ids = encode(text)
    
    with torch.no_grad():
        logits = model(input_ids)
    
    pred_ids = torch.argmax(logits, dim=-1)
    
    decoded = decode(pred_ids)
    
    print('input: '+ text)
    
    print('output: '+ decoded)





    
