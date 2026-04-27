import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import yaml

# modified from https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_seq, head_size, n_embd, msk_attn, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        if msk_attn: self.register_buffer("tril", torch.tril(torch.ones(n_seq, n_seq)))
        self.msk_attn = msk_attn

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        if self.msk_attn:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_seq, n_head, n_embd, msk_attn, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_seq, head_size, n_embd, msk_attn, dropout) \
            for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_seq, n_head, n_embd, msk_attn, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(n_seq, n_head, n_embd, msk_attn, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TTNModel(nn.Module):

    def __init__(
        self, 
        n_embd_in=768, 
        n_seq=11, 
        n_layer=8, 
        n_head=8, 
        n_embd=512, 
        tkn_fc=1024, 
        msk_attn=False, 
        dropout=0, 
        device="mps",
        ):
        super().__init__()

        self.token_embed_setup(n_embd_in, tkn_fc, n_embd, dropout)
        self.position_embedding_table = nn.Embedding(n_seq, n_embd)
        self.blocks = nn.Sequential(*[Block(n_seq, n_head, n_embd, msk_attn, dropout) \
            for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, 1) # Predict in or out of class.

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

        self.device = device
        self.n_embd = n_embd

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def token_embed_setup(self, n_embd_in, tkn_fc, n_embd, dropout):

        if isinstance(tkn_fc, int): tkn_fc = [tkn_fc]
        dims = [n_embd_in] + tkn_fc + [n_embd]

        layers =[]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        self.token_embedding_table = nn.Sequential(*layers)

    def forward(self, x_in, targets=None, pos_weight=1):
        B, T, _ = x_in.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(x_in) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x)[:,:,0] # (B,T)

        # Calculate loss.
        if targets is None:
            loss = None
        else:
            # LIG-based Accept/Reject classification loss.
            if pos_weight == 1:
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            else:
                loss = F.binary_cross_entropy_with_logits(
                    logits, 
                    targets, 
                    pos_weight=torch.tensor(pos_weight, device=self.device)
                )

        return logits, loss 

    def generate_ttn_tokens(self, x_in):
        tok_emb = self.token_embedding_table(x_in) # (B,T,C)
        return tok_emb

def load_ttn_model(model_dir, emb_dim, device):
    print(f"Loading {model_dir} torch model.")
    with open(os.path.join(model_dir, "model.yaml"), "rb") as f:
        mparam = yaml.safe_load(f)
    nembd0, nref, nlayer = emb_dim, mparam["nref"], mparam["nlayer"], 
    nhead, nembd, tkn_fc = mparam["nhead"], mparam["nembd"], mparam["tkn_fc"], 
    msk_attn, dropout = mparam["msk_attn"], mparam["dropout"]
    nseq = mparam["nselect"] + nref
    model = TTNModel(nembd0, nseq, nlayer, nhead, nembd, tkn_fc, msk_attn, dropout, device)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "best.pth"),
        weights_only=True, 
        map_location=torch.device("cpu")
    ))
    m = model.to(device)
    if mparam["input_type"] == "float16": m.half() # float16
    m.eval()
    print(f"Model has {sum(p.numel() for p in m.parameters())/1e6} M parameters.")
    m.nref = nref
    m.nseq = nseq
    m.nembd = nembd
    m.input_type = mparam["input_type"]

    return m

