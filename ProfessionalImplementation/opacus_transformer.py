### Opacus library required: 
##  pip install opacus
#   install other libraries if needed

# "The secret code is" prompt is default here for canary call 



import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from opacus import PrivacyEngine
from opacus.layers.dp_multihead_attention import DPMultiheadAttention
from collections import Counter

# ------------------------
# 1. Hyperparameters
# ------------------------

# Tested parameters with good outcomes 
# batch_size       = 32
# seq_len          = 30        
# embed_size       = 512       
# nhead            = 8
# hidden_dim       = 512
# nlayers          = 4
# dropout          = 0.1
# learning_rate    = 1e-3      
# epochs           = 20
# max_grad_norm    = 1.0
# noise_multiplier = 1
# target_delta     = 1e-3
# vocab_max_size   = 10000     
# min_word_freq    = 1      

# batch_size       = 64
# seq_len          = 128  # = same as block_size        
# embed_size       = 256       
# nhead            = 8
# hidden_dim       = 512
# nlayers          = 4
# dropout          = 0.2
# learning_rate    = 1e-3      
# epochs           = 50
# max_grad_norm    = 1.0
# noise_multiplier = 2
# target_delta     = 1e-4
# vocab_max_size   = 10000     
# min_word_freq    = 1         

batch_size       = 16
seq_len          = 32  # = same as block_size        
embed_size       = 64      
nhead            = 4
hidden_dim       = 128
nlayers          = 4
dropout          = 0.2
learning_rate    = 1e-3      
epochs           = 50
max_grad_norm    = 1.0
noise_multiplier = 2
target_delta     = 1e-4
vocab_max_size   = 10000     
min_word_freq    = 1       

torch.manual_seed(2620)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------
# 2. Read and tokenize text
# ------------------------
with open("seuss_works.txt", "r", encoding="utf-8") as f:
    text = f.read()

def tokenize_to_words(text):
    pattern = r'(\b\w+\b|[^\w\s]+|\s+)'
    tokens = re.findall(pattern, text.lower())
    return [t for t in tokens if t == '\n' or t.strip()]

words = tokenize_to_words(text)
print(f"Total number of word tokens: {len(words)}")

word_counts = Counter(words)
print(f"Total unique words (including newline): {len(word_counts)}")

special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "\n"]
word_list = [
    w for w, cnt in word_counts.most_common(vocab_max_size - len(special_tokens))
    if cnt >= min_word_freq and w != "\n"
]
vocab = special_tokens + word_list

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

UNK_IDX = word2idx["<UNK>"]
PAD_IDX = word2idx["<PAD>"]

vocab_size = len(vocab)
print(f"Final vocabulary size: {vocab_size}")

word_indices = [word2idx.get(word, UNK_IDX) for word in words]
data = torch.tensor(word_indices, dtype=torch.long)

sequences = []
for i in range(0, len(data) - seq_len, seq_len):
    seq = data[i : i + seq_len + 1]
    sequences.append(seq)
print(f"Created {len(sequences)} total sequences")

# ------------------------
# 3. Dataset split
# ------------------------
class TextDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return seq[:-1], seq[1:]

full_dataset = TextDataset(sequences)
train_size = int(0.9 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)

# ------------------------
# 4. Positional Encoding
# ------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

# ------------------------
# 5. DP Transformer Encoder Layer
# ------------------------
class DPTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = DPMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=False
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, is_causal=False, src_key_padding_mask=None):
        attn_out, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src2 = self.dropout1(attn_out)
        src = self.norm1(src + src2)
        ff = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        src = self.norm2(src + ff)
        return src

# ------------------------
# 6. Full Transformer Model
# ------------------------
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nhid, nlayers, dropout=0.1):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = DPTransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.decoder = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self._init_weights()

    def _init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src, src_mask)
        return self.decoder(out)

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

# ------------------------
# 7. Instantiate model & print counts
# ------------------------
model = TransformerModel(
    vocab_size, embed_size, nhead, hidden_dim, nlayers, dropout
).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

print(f"Total parameters: {count_parameters(model)}")

# ------------------------
# 8. Make DP private
# ------------------------
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=noise_multiplier,
    max_grad_norm=max_grad_norm,
)

# ------------------------
# 9. Training & Validation Loop
# ------------------------
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for src_batch, tgt_batch in train_loader:
        src = src_batch.t().to(device)
        tgt = tgt_batch.t().to(device)
        optimizer.zero_grad()
        mask = generate_square_subsequent_mask(src.size(0)).to(device)
        output = model(src, mask)
        loss = criterion(output.view(-1, vocab_size), tgt.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for src_batch, tgt_batch in val_loader:
            src = src_batch.t().to(device)
            tgt = tgt_batch.t().to(device)
            mask = generate_square_subsequent_mask(src.size(0)).to(device)
            output = model(src, mask)
            loss = criterion(output.view(-1, vocab_size), tgt.reshape(-1))
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    if noise_multiplier > 0.0:
        try:
            eps = privacy_engine.get_epsilon(delta=target_delta)
        except OverflowError:
            eps = float("inf")
        print(f"Epoch {epoch} | train loss {avg_train_loss:.4f} | val loss {avg_val_loss:.4f} | ε = {eps:.2f}, δ = {target_delta}")
    else:
        print(f"Epoch {epoch} | train loss {avg_train_loss:.4f} | val loss {avg_val_loss:.4f} | ε = ∞ (no noise), δ = {target_delta}")

# ------------------------
# 10. Text Generation
# ------------------------
def generate(model, start_text, max_len=50, temperature=1.0):
    model.eval()
    tokens = tokenize_to_words(start_text.lower())
    input_indices = [word2idx.get(w, UNK_IDX) for w in tokens]
    input_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(1)
    generated = start_text
    for _ in range(max_len):
        seq_len = input_tensor.size(0)
        if seq_len > 100:
            input_tensor = input_tensor[-100:]
            seq_len = 100
        mask = generate_square_subsequent_mask(seq_len).to(device)
        with torch.no_grad():
            out = model(input_tensor, mask)
            logits = out[-1, 0] / temperature
            probs = torch.softmax(logits, dim=0)
            next_idx = torch.multinomial(probs, 1).item()
        next_tok = idx2word.get(next_idx, "<UNK>")
        if next_tok == "\n":
            generated += "\n"
        elif not re.match(r'^[^\w\s]', next_tok):
            generated += " " + next_tok
        else:
            generated += next_tok
        next_tensor = torch.tensor([[next_idx]], device=device)
        input_tensor = torch.cat([input_tensor, next_tensor], dim=0)
    return generated

print("\n--- Generated Text ---\n")
print(generate(model, "The secret code is", max_len=100, temperature=0.8))
# pip install required modules if needed 


import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from opacus import PrivacyEngine
from opacus.layers.dp_multihead_attention import DPMultiheadAttention
from collections import Counter

# ------------------------
# 1. Hyperparameters
# ------------------------

# Tested parameters with good outcomes 
# batch_size       = 32
# seq_len          = 30        
# embed_size       = 512       
# nhead            = 8
# hidden_dim       = 512
# nlayers          = 4
# dropout          = 0.1
# learning_rate    = 1e-3      
# epochs           = 20
# max_grad_norm    = 1.0
# noise_multiplier = 1
# target_delta     = 1e-3
# vocab_max_size   = 10000     
# min_word_freq    = 1      

batch_size       = 64
seq_len          = 128  # = same as block_size        
embed_size       = 256       
nhead            = 8
hidden_dim       = 512
nlayers          = 4
dropout          = 0.2
learning_rate    = 1e-3      
epochs           = 50
max_grad_norm    = 1.0
noise_multiplier = 2
target_delta     = 1e-4
vocab_max_size   = 10000     
min_word_freq    = 1         

torch.manual_seed(2620)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------
# 2. Read and tokenize text
# ------------------------
with open("seuss_works.txt", "r", encoding="utf-8") as f:
    text = f.read()

def tokenize_to_words(text):
    pattern = r'(\b\w+\b|[^\w\s]+|\s+)'
    tokens = re.findall(pattern, text.lower())
    return [t for t in tokens if t == '\n' or t.strip()]

words = tokenize_to_words(text)
print(f"Total number of word tokens: {len(words)}")

word_counts = Counter(words)
print(f"Total unique words (including newline): {len(word_counts)}")

special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "\n"]
word_list = [
    w for w, cnt in word_counts.most_common(vocab_max_size - len(special_tokens))
    if cnt >= min_word_freq and w != "\n"
]
vocab = special_tokens + word_list

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

UNK_IDX = word2idx["<UNK>"]
PAD_IDX = word2idx["<PAD>"]

vocab_size = len(vocab)
print(f"Final vocabulary size: {vocab_size}")

word_indices = [word2idx.get(word, UNK_IDX) for word in words]
data = torch.tensor(word_indices, dtype=torch.long)

sequences = []
for i in range(0, len(data) - seq_len, seq_len):
    seq = data[i : i + seq_len + 1]
    sequences.append(seq)
print(f"Created {len(sequences)} total sequences")

# ------------------------
# 3. Dataset split
# ------------------------
class TextDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return seq[:-1], seq[1:]

full_dataset = TextDataset(sequences)
train_size = int(0.9 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)

# ------------------------
# 4. Positional Encoding
# ------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

# ------------------------
# 5. DP Transformer Encoder Layer
# ------------------------
class DPTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = DPMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=False
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, is_causal=False, src_key_padding_mask=None):
        attn_out, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src2 = self.dropout1(attn_out)
        src = self.norm1(src + src2)
        ff = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        src = self.norm2(src + ff)
        return src

# ------------------------
# 6. Full Transformer Model
# ------------------------
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nhid, nlayers, dropout=0.1):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = DPTransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.decoder = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self._init_weights()

    def _init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src, src_mask)
        return self.decoder(out)

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

# ------------------------
# 7. Instantiate model & print counts
# ------------------------
model = TransformerModel(
    vocab_size, embed_size, nhead, hidden_dim, nlayers, dropout
).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

print(f"Total parameters: {count_parameters(model)}")

# ------------------------
# 8. Make DP private
# ------------------------
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=noise_multiplier,
    max_grad_norm=max_grad_norm,
)

# ------------------------
# 9. Training & Validation Loop
# ------------------------
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for src_batch, tgt_batch in train_loader:
        src = src_batch.t().to(device)
        tgt = tgt_batch.t().to(device)
        optimizer.zero_grad()
        mask = generate_square_subsequent_mask(src.size(0)).to(device)
        output = model(src, mask)
        loss = criterion(output.view(-1, vocab_size), tgt.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for src_batch, tgt_batch in val_loader:
            src = src_batch.t().to(device)
            tgt = tgt_batch.t().to(device)
            mask = generate_square_subsequent_mask(src.size(0)).to(device)
            output = model(src, mask)
            loss = criterion(output.view(-1, vocab_size), tgt.reshape(-1))
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    if noise_multiplier > 0.0:
        try:
            eps = privacy_engine.get_epsilon(delta=target_delta)
        except OverflowError:
            eps = float("inf")
        print(f"Epoch {epoch} | train loss {avg_train_loss:.4f} | val loss {avg_val_loss:.4f} | ε = {eps:.2f}, δ = {target_delta}")
    else:
        print(f"Epoch {epoch} | train loss {avg_train_loss:.4f} | val loss {avg_val_loss:.4f} | ε = ∞ (no noise), δ = {target_delta}")

# ------------------------
# 10. Text Generation
# ------------------------
def generate(model, start_text, max_len=50, temperature=1.0):
    model.eval()
    tokens = tokenize_to_words(start_text.lower())
    input_indices = [word2idx.get(w, UNK_IDX) for w in tokens]
    input_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(1)
    generated = start_text
    for _ in range(max_len):
        seq_len = input_tensor.size(0)
        if seq_len > 100:
            input_tensor = input_tensor[-100:]
            seq_len = 100
        mask = generate_square_subsequent_mask(seq_len).to(device)
        with torch.no_grad():
            out = model(input_tensor, mask)
            logits = out[-1, 0] / temperature
            probs = torch.softmax(logits, dim=0)
            next_idx = torch.multinomial(probs, 1).item()
        next_tok = idx2word.get(next_idx, "<UNK>")
        if next_tok == "\n":
            generated += "\n"
        elif not re.match(r'^[^\w\s]', next_tok):
            generated += " " + next_tok
        else:
            generated += next_tok
        next_tensor = torch.tensor([[next_idx]], device=device)
        input_tensor = torch.cat([input_tensor, next_tensor], dim=0)
    return generated

print("\n--- Generated Text ---\n")
print(generate(model, "The secret code is", max_len=100, temperature=0.8))
