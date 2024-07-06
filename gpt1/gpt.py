import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
torch.manual_seed(1337)
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
import wandb

device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print('GPU:', torch.cuda.get_device_properties(0).name)
else:
    print("CPU")

with open("./tinyShakesphere.txt", "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(set(text))
vocab_size = len(chars)
print(''.join(chars))
print(len(chars))

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

encoded_text = torch.tensor(encode(text), dtype=torch.long).to(device)

train_size = 0.9

train_data = encoded_text[:int(train_size*len(encoded_text))]
val_data = encoded_text[int(train_size*len(encoded_text)):]

class GPTConfig:
    vocab_size: int = None
    batch_size: int = 8
    block_size: int = 16
    n_embed: int = 128 
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.2
    bias: bool = False

    max_iters: int = 10000
    eval_interval: int = 500
    lr: float = 1e-3
    eval_iters: int = 100
    
   

def get_batch(split, config):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    return x.to(device), y.to(device)

class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0

        self.key = nn.Linear(config.n_embed, config.n_embed // config.n_heads, bias=config.bias)
        self.query = nn.Linear(config.n_embed, config.n_embed // config.n_heads, bias=config.bias)
        self.value = nn.Linear(config.n_embed, config.n_embed // config.n_heads, bias=config.bias)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size, device=device)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out
    
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_embed, config.n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed, bias= config.bias),
            nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed, bias= config.bias),
            nn.Dropout(config.dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.atn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.atn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
        
    def __init__(self,config):
        super().__init__()
        assert config.vocab_size is not None, "vocab_size is not set"
        assert config.block_size is not None, "block_size is not set"
        assert config.n_embed is not None, "n_embed is not set"
        assert config.n_heads is not None, "n_heads is not set"
        assert config.n_layers is not None, "n_layers is not set"
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed),
            position_embedding_table = nn.Embedding(config.vocab_size, config.n_embed),
            drouput = nn.Dropout(config.dropout),
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embed, bias = config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
        
        print("No of parameters: ", sum(p.numel() for p in self.parameters()))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.position_embedding_table.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        B,T = idx.size()

        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (T)


        tok_emb = self.transformer.token_embedding_table(idx) # (B,T, n_embed)
        pos_emb = self.transformer.position_embedding_table(torch.arange(T, device=device)) #(T, n_embed)

        x = tok_emb + pos_emb # (B,T, C) 
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)

        
        if targets is None:
            logits =  self.lm_head(x) # (B,T, vocab_size)
            loss = None
        else:
            logits =  self.lm_head(x)
            B, T, C =  logits.size()
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

config = GPTConfig()
config.vocab_size = vocab_size
gpt = GPT(config=config).to(device)
# gpt = torch.compile(gpt)
wandb.init(
    # set the wandb project where this run will be logged
    project="GPT",

    # track hyperparameters and run metadata
    config={
    "learning_rate": config.lr,
    "architecture": "Tranformer",
    "dataset": "tinyShakesphere",
    'n_head': config.n_heads,
    'batch_size': config.batch_size,
    'block_size': config.block_size,
    'n_embed': config.n_embed,
    'n_layers': config.n_layers,
    'dropout': config.dropout,
    "epochs": 1,
    }
)
print(sum(p.numel() for p in gpt.parameters()))

optimizer = optim.AdamW(gpt.parameters(), lr=config.lr)

@torch.no_grad()
def estimate_loss(model, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split, config)
            logits, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

train_loss = []
val_loss = []
step = []

for steps in tqdm(range(config.max_iters)):
    if steps % config.eval_iters == 0:
        losses = estimate_loss(gpt, config.eval_iters)
        print(f'step {steps}: train loss {losses["train"]:.4f}, test loss {losses["val"]:.4f}')
        wandb.log({
            "step": steps,
            "train_loss": losses["train"],
            "test_loss": losses["val"],
        })
        train_loss.append(losses['train'])
        val_loss.append(losses['val'])
        step.append(step[-1] + config.eval_iters if step else config.eval_iters)
    xb, yb = get_batch('train', config)
    logits, loss = gpt(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

plt.plot(step, train_loss, label='Train Loss')
plt.plot(step, val_loss, label='Test Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()

# print(decode(gpt.generate(torch.tensor([[0]], dtype=torch.long, device=device), max_new_tokens=1000)[0].tolist()))

with open('generated.txt', 'w') as f:
    f.write(decode(gpt.generate(torch.tensor([[0]], dtype=torch.long, device=device), max_new_tokens=10000)[0].tolist()))

wandb.finish()