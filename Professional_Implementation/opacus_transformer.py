### Opacus library required: 
##  pip install opacus
#   install other libraries if needed

import math
import re
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from opacus import PrivacyEngine
from opacus.layers.dp_multihead_attention import DPMultiheadAttention

# ------------------------
# 1. Hyperparameters
# ------------------------
batch_size       = 16
seq_len          = 8
embed_size       = 32
nhead            = 4
hidden_dim       = embed_size * 4   # feed-forward dimension = 4×embed_size
nlayers          = 4
dropout          = 0.2
learning_rate    = 1e-3
epochs           = 100
max_grad_norm    = 1.0
target_delta     = 1e-4
target_epsilon   = 10

# Seed for reproducibility 
torch.manual_seed(2620)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}\n")

# ------------------------
# 1b. CSV logger setup
# ------------------------
log_path = "training_log.csv"
log_file = open(log_path, mode="w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["epoch", "train_loss", "val_loss", "epsilon"])

# ------------------------
# 2. Read, tokenize & encode text
# ------------------------
with open("seuss_works.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

def splitter(text: str) -> list[str]:
    pattern = r'(\w+|[^\w\s]|\s|\n)'
    tokens = re.findall(pattern, text)
    formatted: list[str] = []
    for tok in tokens:
        if tok.istitle():
            formatted.append("<C>")
            formatted.append(tok.lower())
        elif tok.isupper():
            formatted.append("<A>")
            formatted.append(tok.lower())
        else:
            formatted.append(tok)
    return formatted

formatted     = splitter(raw_text)
unique_tokens = sorted(set(formatted))
token2idx     = {tok: i for i, tok in enumerate(unique_tokens)}
idx2token     = {i: tok for tok, i in token2idx.items()}
all_indices   = torch.tensor([token2idx[t] for t in formatted], dtype=torch.long)

# ------------------------
# 3. Train/Validation split & DataLoader
# ------------------------
n_train     = int(0.9 * len(all_indices))
train_data  = all_indices[:n_train]
val_data    = all_indices[n_train:]

class WindowDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

train_ds     = WindowDataset(train_data, seq_len)
val_ds       = WindowDataset(val_data,   seq_len)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

# ------------------------
# 4. DP Transformer Encoder Layer (pre-norm)
# ------------------------
class DPTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn  = DPMultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=False)
        self.linear1    = nn.Linear(d_model, dim_feedforward)
        self.linear2    = nn.Linear(dim_feedforward, d_model)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, is_causal=False, src_key_padding_mask=None):
        # Pre-norm self-attention
        x = src
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=src_mask)
        x = x + self.dropout1(attn_out)

        # Pre-norm feed-forward
        x_norm2 = self.norm2(x)
        ff = self.linear2(F.relu(self.linear1(x_norm2)))
        x = x + self.dropout2(ff)
        return x

# ------------------------
# 5. Full Transformer Model (learned pos embeddings)
# ------------------------
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nhid, nlayers, dropout=0.1):
        super().__init__()
        self.encoder             = nn.Embedding(vocab_size, d_model)
        self.position_embedding  = nn.Embedding(seq_len, d_model)
        encoder_layer            = DPTransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.decoder             = nn.Linear(d_model, vocab_size)
        self.d_model             = d_model

    def forward(self, src, src_mask):
        # src: (seq_len, batch_size)
        tok_emb = self.encoder(src) * math.sqrt(self.d_model)           # (seq_len, batch, d_model)
        pos_idx = torch.arange(src.size(0), device=src.device)         # (seq_len,)
        pos_emb = self.position_embedding(pos_idx)                     # (seq_len, d_model)
        x = tok_emb + pos_emb.unsqueeze(1)                             # (seq_len, batch, d_model)
        out = self.transformer_encoder(x, src_mask)
        return self.decoder(out)

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

# ------------------------
# 6. Instantiate, optimizer, criterion
# ------------------------
vocab_size = len(unique_tokens)
model      = TransformerModel(vocab_size, embed_size, nhead, hidden_dim, nlayers, dropout).to(device)

# ← Use SGD optimizer
optimizer  = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

criterion  = nn.CrossEntropyLoss()

param_count = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {param_count}\n")

# ------------------------
# 7. Privacy engine
# ------------------------

# 'with_epsilon' variation used to reach set privacy budget
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    max_grad_norm=max_grad_norm,
    target_epsilon=target_epsilon,
    target_delta=target_delta,
    epochs=epochs
)

# ------------------------
# 9. Training & Validation Loop
# ------------------------
for epoch in range(1, epochs + 1):
    # Train
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.t().to(device)
        yb = yb.t().to(device)
        optimizer.zero_grad()
        mask   = generate_square_subsequent_mask(xb.size(0)).to(device)
        logits = model(xb, mask)
        loss   = criterion(logits.view(-1, vocab_size), yb.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train = total_loss / len(train_loader)

    # Validate 
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.t().to(device)
            yb = yb.t().to(device)
            mask   = generate_square_subsequent_mask(xb.size(0)).to(device)
            logits = model(xb, mask)
            loss   = criterion(logits.view(-1, vocab_size), yb.reshape(-1))
            val_loss += loss.item()
    avg_val = val_loss / len(val_loader)

    # DP budget 
    try:
        eps = privacy_engine.get_epsilon(delta=target_delta)
    except OverflowError:
        eps = float('inf')

    # Print and log 
    print(f"Epoch {epoch} | train: {avg_train:.4f} | val: {avg_val:.4f} | ε={eps:.2f}, δ={target_delta}")
    log_writer.writerow([epoch, f"{avg_train:.4f}", f"{avg_val:.4f}", f"{eps:.2f}"])

# ------------------------
# 10. Cleanup CSV
# ------------------------
log_file.close()

# ------------------------
# 11. Text Generation
# ------------------------
def generate(model, start_text, max_len=50, temperature=1.0):
    model.eval()
    tokens       = splitter(start_text)
    idxs         = [token2idx.get(tok, 0) for tok in tokens]
    input_tensor = torch.tensor(idxs, dtype=torch.long, device=device).unsqueeze(1)
    out_text     = start_text

    for _ in range(max_len):
        sz = input_tensor.size(0)
        if sz > seq_len:
            input_tensor = input_tensor[-seq_len:]
            sz = seq_len
        mask = generate_square_subsequent_mask(sz).to(device)
        with torch.no_grad():
            logits = model(input_tensor, mask)
            logits = logits[-1, 0] / temperature
            probs  = F.softmax(logits, dim=-1)
            nxt    = torch.multinomial(probs, 1).item()
        tok = idx2token[nxt]
        if tok == "\n":
            out_text += "\n"
        elif tok.isspace() or re.match(r'^[^\w\s]', tok):
            out_text += tok
        else:
            out_text += tok
        next_tensor  = torch.tensor([[nxt]], device=device)
        input_tensor = torch.cat([input_tensor, next_tensor], dim=0)

    return out_text

if __name__ == "__main__":
    print("\n--- Generated Text ---\n")
    print(generate(model, "The cat in the hat", max_len=50, temperature=1))
