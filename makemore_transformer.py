import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import random

block_size = 32
batch_size = 128
num_steps = 5000
eval_interval = 250
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
embd_size = 128
num_heads = 4
num_layers = 4
dropout = 0.2


with open('names.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
ctoi = {ch: i for i, ch in enumerate(chars)}
itoc = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)
encode = lambda s: [ctoi[c] for c in s]
decode = lambda l: ''.join([itoc[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embd_size, head_size, bias=False)
        self.query = nn.Linear(embd_size, head_size, bias=False)
        self.value = nn.Linear(embd_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        if mask is not None:
            wei = wei + mask
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embd_size, embd_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, embd_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_size, 4 * embd_size),
            nn.ReLU(),
            nn.Linear(4 * embd_size, embd_size),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embd_size, num_heads):
        super().__init__()
        head_size = embd_size // num_heads
        self.sa = MultiHeadSelfAttention(num_heads, head_size)
        self.ffwd = FeedForward(embd_size)
        self.ln1 = nn.LayerNorm(embd_size)
        self.ln2 = nn.LayerNorm(embd_size)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embd_size)
        self.position_embedding_table = nn.Embedding(block_size, embd_size)
        self.blocks = nn.ModuleList([TransformerBlock(embd_size, num_heads=num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embd_size)
        self.lm_head = nn.Linear(embd_size, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _generate_causal_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        
        mask = self._generate_causal_mask(T, device=x.device)
        
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

# train and eval
model = Transformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print(f"Starting training on {device} with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters... 🚀")

for step in range(num_steps):
    if step % eval_interval == 0 or step == num_steps - 1:
        losses = estimate_loss()
        print(f"Step {step:6d}/{num_steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


start_context = torch.tensor([[ctoi['\n']]], dtype=torch.long, device=device)
for _ in range(10):
    generated_indices = model.generate(start_context, max_new_tokens=20)[0].tolist()
    generated_name = decode(generated_indices).split('\n')[1]
    print(generated_name)
