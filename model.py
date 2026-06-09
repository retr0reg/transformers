from math import sqrt
import torch
import torch.nn as nn
from torch.nn import ReLU, functional as F

# do not give me complete code, only tabs from same line

def get_device():
    return 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# hyperparameters

block_size = 8
# NOTE: when 8 individuals, there are only 7 combinations (since a token can't be by itself)
# NOTE: when setting a block_size of n, that means we need to truncate after n, since the machine never looked at anything more than n
batch_size = 4 # how many blocks we will deal with at the same time?
n_embd     = 32

max_iters     = 5000
learning_rate = 1e-3
eval_iters    = 200

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        # K, Q, V
        self.key = nn.Linear(n_embd, head_size, bias=False)      # bias=False to only conduct matrix multiplications
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # tril is a variable, a "buffer" in torch
        self.register_buffer('tril', torch.tril(
            torch.ones(
                block_size, block_size # bck_sz * bck_sz
            )
        ))

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        k: torch.Tensor = self.key(x)
        q: torch.Tensor = self.query(x)

        # attention score
        qk = q @ k.transpose(-2, -1) # dot product (B, T, C) * (B, C, T) -> (B, T, T)
        qk *= 1/sqrt(k.size(-1)) # scaled attension to keep softmax diffused but saturated
        qk = qk.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # no future tokens
        qk = F.softmax(qk, dim=-1) # softmax on last axis: for a given query, softmax of keys

        v: torch.Tensor = self.value(x)

        return qk @ v

class MultiHead(nn.Module):

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)]) # list like
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1) # pipeline like
        x = self.proj(x)
        return x

class FeedForward(nn.Module): # (per-token) process token itself (updated info). (+ factual info?)
    
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            ReLU(),
            nn.Linear(n_embd * 4, n_embd), # project, back to residual connection
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    
    def __init__(self, n_embd, n_head) -> None:
        super().__init__()
        head_size = n_embd // n_head

        self.attention_heads = MultiHead(n_head, head_size)
        self.feedforward     = FeedForward(n_embd)
        self.layernorm_1     = nn.LayerNorm(n_embd)
        self.layernorm_2     = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention_heads(self.layernorm_1(x)) # x+ -> residual connnection, gradient highway
        x = x + self.feedforward(self.layernorm_2(x))
        return x


class LM(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd),
        )
        self.lm_head         = nn.Linear(n_embd, vocab_size) # decoder language model head (latent to scoring on vocab)

    def forward(self, idx: torch.Tensor, target=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=get_device()))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # guess
            target = target.view(B*T)    # answer
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T)
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -block_size:] # crop to fit within block_size, for the sake of position_embd_table
            
            logits, loss = self(idx_crop) # get prediction
            logits = logits[:, -1, :] # only get the logits for final token, become (B, C)
            probs  = F.softmax(logits, dim=-1)

            _next = torch.multinomial(probs, num_samples=1) # sample a (predict) next token from probs
            idx = torch.cat((idx, _next), dim=1) # make the sequence, (B, T+1)
        
        return idx



if __name__ == "__main__":

    with open('data/pg64317.txt', 'r') as fp:
        text = fp.read()

    characters = sorted(list(set(text)))
    vocab_size = len(characters)

    # token (characters)
    char_to_int = { chr:i for i, chr in enumerate(characters)}
    int_to_char = { i:chr for i, chr in enumerate(characters)}

    # encoding and decoding (strings):
    encode = lambda string: [char_to_int[char] for char in string]
    decode = lambda ints: [int_to_char[i] for i in ints]

    data = torch.tensor(encode(text), dtype=torch.long)

    # spliting into train and validate
    spliter = int(0.9*len(data))
    train = data[:spliter]
    validate = data[spliter:]

    # TRAIN
    model = LM()
    m = model.to(get_device()) # mac m-chip
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def get_batch(split):
        data = train if split == 'train' else validate
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(get_device()), y.to(get_device())
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'validate']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    
    # first loss
    # statistically speaking we it should be around -ln(1/45) (vocab_size, possible next token)

    for iter in range(max_iters):
        
        # eval on iterations
        if iter % eval_iters == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, validate loss {losses['validate']:.4f}")

        xb, yb = get_batch('train')
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=get_device())
    # batch=1, time=1, 1x1 tensor holding zeros
    # zero is the token_id for a newline '\n' chatacter
    print(''.join(decode(model.generate(context, max_new_tokens=2000)[0].tolist())))
