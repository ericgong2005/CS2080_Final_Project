import torch
import torch.nn as nn
from torch.nn import functional as F

import re
import os
import matplotlib.pyplot as plt
import math
import string
import csv, pathlib, random
from datetime import datetime
from optim import DPSGD
RUN_TAG = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 


# hyperparameters
# batch_size = 64
# block_size = 128
# iterations = 2000
# iteration_checkpoint = 20
# learning_rate = 1e-6
# device = "mps" if torch.backends.mps.is_available() else 'cpu' # For MacOS GPU acceleration
# loss_evaluation_iterations = 64
# embedding_count = 256
# head_count = 4
# layer_count = 4
# dropout_rate = 0.2
batch_size = 16
block_size = 8
iterations = 100
iteration_checkpoint = 1
learning_rate = 1e-2
device = "mps" if torch.backends.mps.is_available() else 'cpu' # For MacOS GPU acceleration
loss_evaluation_iterations = 4
embedding_count = 64
head_count = 4
layer_count = 4
dropout_rate = 0.2

torch.manual_seed(1234)

print(f"Using {device}\n")

def pretty_join(tokens: list[str]) -> str:
    """Insert spaces between tokens where needed and tidy punctuation."""
    out = []
    for tok in tokens:
        if tok.isspace():
            out.append(tok)
        elif tok in string.punctuation:
            if out and out[-1].endswith(' '):
                out[-1] = out[-1].rstrip()
            out.append(tok)
        else:                           # ordinary word
            if out and not out[-1].endswith((' ', '\n')):
                out.append(' ')
            out.append(tok)
    return ''.join(out)

def apply_case(tokens: list[str]) -> list[str]:
    """Handle <C>/<A> markers produced by splitter()."""
    out, mode = [], 0                  # 0 = normal, 1 = capitalise, 2 = upper
    for t in tokens:
        if t == "<C>":
            mode = 1
        elif t == "<A>":
            mode = 2
        else:
            if mode == 1:
                out.append(t.capitalize())
            elif mode == 2:
                out.append(t.upper())
            else:
                out.append(t)
            mode = 0
    return out

# Import the text for training the model
with open('seuss_works.txt', 'r', encoding='utf-8') as f:
    training_text = f.read()

def splitter(text : str) -> list[str]:
    pattern = r'(\w+|[^\w\s]|\s|\n)'
    split = re.findall(pattern, text)
    formatted = []
    for word in split:
        if word.istitle():
            formatted.append("<C>")
            formatted.append(word.lower())
        elif word.isupper():
            formatted.append("<A>")
            formatted.append(word.lower())
        else:
            formatted.append(word)
    return formatted

formatted = splitter(training_text)

unique = set(formatted)

training_data = formatted
tokens = unique

# Defining an Encoder and Decoder
sorted_tokens = sorted(tokens)
encode_dict = {element: idx for idx, element in enumerate(sorted_tokens)}
decode_dict = {idx: element for idx, element in enumerate(sorted_tokens)}

def encoder(text: list[str]) -> list[int] :
    return [encode_dict[element] for element in text]

def decoder(code: list[int]) -> list[str] :
    return [decode_dict[element] for element in code]

def full_decode(code: list[str]) -> str :
    status = 0
    final = ""
    for element in code:
        if element == "<C>":
            status = 1
        elif element == "<A>":
            status = 2
        else:
            if status == 0:
                final += element
            elif status == 1:
                final += element.capitalize()
            else:
                final += element.upper()
            status = 0
    return final

# Training and Validation Split
data = torch.tensor(encoder(training_data), dtype=torch.long)
n = int(0.9*len(data))
training = data[:n]
validation = data[n:]
vocab_size = len(tokens)

# data loading
def get_batch(type: str):
    # generate a batch of inputs x and targets y
    data = training if type == 'training' else validation
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['training', 'validation']:
        losses = torch.zeros(loss_evaluation_iterations)
        for k in range(loss_evaluation_iterations):
            X, Y = get_batch(split)
            _logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# One self-attention head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_count, head_size, bias=False)
        self.query = nn.Linear(embedding_count, head_size, bias=False)
        self.value = nn.Linear(embedding_count, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # input (Batch x Time x Channels)
        # output (Batch x Time x Head)
        B,T,C = x.shape
        k = self.key(x) # (Batch x Time x Head)
        q = self.query(x) # (Batch x Time x Head)
        v = self.value(x) # (Batch x Time x Head)
        
        # Calculating the weights via dot-product to facilitate interaction
        weights =  q @ k.transpose(-2, -1) # (Batch x Time x Head) @ (Batch x Head x Time) = (Batch x Time x Time)
        
        # Normalizing the weights to have variance close to 1, to prevent softmax from overweighing the max
        weights = weights * head_count**-0.5
        
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        # weighted aggregation of values
        output = weights @ v # (Batch x Time x Time) @ (Batch x Time x Head) = (Batch x Time x Head)
        return output

# Execute multiple self-attention heads in parallel
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, embedding_count)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

# Feed Forward after self-attention
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # 4x increase in dimension as per "Attention is all you need" Paper parameters
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, embedding_count, head_count):
        super().__init__()
        head_size = embedding_count // head_count
        self.self_attention = MultiHeadAttention(head_count, head_size)
        self.feed_Forward = FeedFoward(embedding_count)
        self.layer_normalization_1 = nn.LayerNorm(embedding_count)
        self.layer_normalization_2 = nn.LayerNorm(embedding_count)

    def forward(self, x):
        # Utilize Residual Connections
        x = x + self.self_attention(self.layer_normalization_1(x))
        x = x + self.feed_Forward(self.layer_normalization_2(x))
        return x

class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_count)
        self.position_embedding_table = nn.Embedding(block_size, embedding_count)
        self.blocks = nn.Sequential(*[Block(embedding_count, head_count) for _ in range(layer_count)])
        self.ln_f = nn.LayerNorm(embedding_count) # final layer normalization
        self.lm_head = nn.Linear(embedding_count, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (Batch x Time) tensor of integers
        token_embeddings = self.token_embedding_table(idx) # (Batch x Time x Channels)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # (Time x Channels)
        x = token_embeddings + position_embeddings # (Batch x Time x Channels)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (Batch x Time x vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (Batch x Time) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (Batch x Channels)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (Batch x Channels)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (Batch x 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (Batch x Time + 1)
        return idx

model = TransformerLanguageModel()
m = model.to(device)

# print the number of parameters in the model
parameter_count = sum(p.numel() for p in m.parameters())
print("Number of parameters: ", parameter_count, "\n")

# create a PyTorch optimizer
#optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


def sigma_from_epsilon(eps, delta, q, T):
    # Moments-accountant bound, roughly tight (c≈1.2 from Abadi et al.)
    c = 1.12
    return c * q * math.sqrt(T * math.log(1/delta)) / eps

q = batch_size / len(data)        # sampling ratio
T = iterations                    # total optimizer steps
target_epsilons = [50]
delta = 1e-4

train_losses = []
val_losses = []
iteration_steps = []

def run_one_trial(eps, trial_id=1):
    # 1  reset the model and RNG
    torch.manual_seed(1234 + trial_id)
    random.seed(1234 + trial_id)
    model.apply(model._init_weights)        # re-initialise weights

    # 2  compute sigma and create a fresh optimiser
    sigma = sigma_from_epsilon(
        eps, delta=1e-4,
        q=batch_size / len(data),
        T=iterations,
    )
    print(f"ε={eps:<4} σ={sigma:.6f}")          # ← prints σ to console
    optimizer = DPSGD(
        named_params=list(model.named_parameters()),
        lot_size=batch_size,
        lr=learning_rate,
        noise_scale=sigma,      
        max_grad_norm=1.0,
        weight_decay=0.0
    )

    # 3  prepare logging
    #csv_path = pathlib.Path("runs") / f"eps{eps}_trial{trial_id}.csv"
    csv_path = pathlib.Path("runs") / f"eps{eps}_trial{trial_id}_{RUN_TAG}.csv"
    csv_path.parent.mkdir(exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "epsilon"])

        train_losses.clear(); val_losses.clear(); iteration_steps.clear()

        # 4  train
        for it in range(iterations):
            if it % iteration_checkpoint == 0:
                losses = estimate_loss()
                writer.writerow([it,
                                 losses["training"].item(),
                                 losses["validation"].item(),
                                 eps])
                print(f"ε={eps} it={it}: "
                      f"train {losses['training']:.4f}  "
                      f"val  {losses['validation']:.4f}")
                iteration_steps.append(it)
                train_losses.append(losses["training"].item())
                val_losses.append(losses["validation"].item())


            # ---- compute per-sample gradients ----
            xb, yb = get_batch('training')
            per_sample_grads = {
                name: [] for name, p in model.named_parameters() if p.requires_grad
            }
            for i in range(xb.size(0)):
                model.zero_grad(set_to_none=True)
                xi = xb[i].unsqueeze(0)
                yi = yb[i].unsqueeze(0)
                _, loss = model(xi, yi)
                loss.backward()
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        per_sample_grads[name].append(p.grad.detach().clone())
            per_sample_grads = {
                name: torch.stack(g) for name, g in per_sample_grads.items()
            }

            optimizer.step(per_sample_grads)

        # 5  save loss-curve PNG
        plt.figure(figsize=(8, 5))
        plt.plot(iteration_steps, train_losses, label="train")
        plt.plot(iteration_steps, val_losses, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"Loss curve – ε={eps}")
        plt.legend()
        plt.grid(True)
        plt.savefig(csv_path.with_suffix(".png"))
        plt.close()


    # 6  generate a short sample **for this ε-specific model**
    start_string = "The cat in the hat"
    context = torch.tensor(
        encoder(splitter(start_string)),
        dtype=torch.long, device=device
    ).unsqueeze(0)

    raw_tokens   = decoder(model.generate(context, max_new_tokens=100)[0].tolist())
    tokens_cased = apply_case(raw_tokens)
    generation   = pretty_join(tokens_cased)

    ts = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    gen_path = pathlib.Path("Transformer_model_generations") \
                   / f"generation_eps{eps}_trial{trial_id}_{ts}.txt"
    gen_path.parent.mkdir(exist_ok=True)
    with gen_path.open("w") as g:
        g.write(generation)

    print(f"\nSample for ε={eps} (saved to {gen_path}):\n{generation[:300]}...\n")


# ---- MAIN LOOP ----
for eid, eps in enumerate([1, 10, 20, 50], start=1):
    run_one_trial(eps, trial_id=eid)


# #THIS FUNCTION IS LLM EDITED!
# for iter in range(iterations):

#     # Periodically evaluate the loss on train and val sets
#     if iter % iteration_checkpoint == 0:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['training']:.4f}, val loss {losses['validation']:.4f}")
#         train_losses.append(losses["training"])
#         val_losses.append(losses["validation"])
#         iteration_steps.append(iter)

#     # sample a batch of data
#     xb, yb = get_batch('training')

#     # compute per-sample gradients
#     per_sample_grads = {name: [] for name, param in model.named_parameters() if param.requires_grad}
#     for i in range(xb.size(0)):
#         model.zero_grad(set_to_none=True)
#         xi = xb[i].unsqueeze(0)
#         yi = yb[i].unsqueeze(0)
#         _, loss = model(xi, yi)
#         loss.backward()
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 per_sample_grads[name].append(param.grad.detach().clone())
#     per_sample_grads = {name: torch.stack(grads) for name, grads in per_sample_grads.items()}

#     # step with DPSGD
#     optimizer.step(per_sample_grads)


# # Plot the Training and Validation Loss
# current_time = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
# plt.figure(figsize=(10, 6))
# plt.plot(iteration_steps, train_losses, label='Training Loss')
# plt.plot(iteration_steps, val_losses, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# hyperparameters_text = (f"Properties:\n"
#                         f"Batch size: {batch_size}\n"
#                         f"Block size: {block_size}\n"
#                         f"Iterations: {iterations}\n"
#                         f"Learning rate: {learning_rate}\n"
#                         f"Embeddings: {head_count}\n"
#                         f"Heads: {iterations}\n"
#                         f"Layers: {layer_count}\n"
#                         f"Parameters: {parameter_count}\n"
#                         f"Device: {device}\n")

# plt.text(0.95, 0.95, hyperparameters_text, transform=plt.gca().transAxes,
#          fontsize=10, verticalalignment='top', horizontalalignment='right',
#          bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# os.makedirs("Transformer_model_loss_plots", exist_ok=True)
# plt.savefig(f"Transformer_model_loss_plots/transformer_loss_{current_time}.png")
# plt.close()

# # generate from the model
# start_string = "The cat in the hat"

# context = torch.tensor(encoder(splitter(start_string)), dtype=torch.long, device=device).unsqueeze(0)

# generation = full_decode(decoder(m.generate(context, max_new_tokens=100)[0].tolist()))

# print(generation)

# current_time = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
# os.makedirs("Transformer_model_generations", exist_ok=True)
# with open(f"Transformer_model_generations/transformer_generation_{current_time}.txt", "w") as f:
#     f.write((f"PROPERTIES:\n"
#                         f"\tBatch size: {batch_size}\n"
#                         f"\tBlock size: {block_size}\n"
#                         f"\tIterations: {iterations}\n"
#                         f"\tLearning rate: {learning_rate}\n"
#                         f"\tEmbeddings: {head_count}\n"
#                         f"\tHeads: {iterations}\n"
#                         f"\tLayers: {layer_count}\n"
#                         f"\tParameters: {parameter_count}\n"
#                         f"\tDevice: {device}\n\n"))
#     f.write("START OF GENERATION:")
#     f.write(generation)