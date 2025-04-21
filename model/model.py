import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # TODO: create a causal mask for attention matrix of shape [config.block_size, config.block_size] (config.block_size is the maximum sequence length)
        #   The matrix should has 1s in the lower left triangular part (including the diagonal) and 0s in the upper right.
        #   Name the matrix `causal_mask` and then expand the mask for the batch and head dimensions
        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size))
        causal_mask = causal_mask.view(1, 1, config.block_size, config.block_size)

        # your code ends here
        
        # register the mask as a buffer so it's not updated as a model parameter
        # but can still be used in the forward pass & saved to the state_dict
        self.register_buffer("causal_mask", causal_mask)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd) (b, n, d)

        # TODO: implement the forward pass of the casual self-attention layer.
        #d = C: dimensionality
        query_projected = self.c_attn(x)
        Q, K, V = query_projected.chunk(3, -1)

        attention = torch.matmul(Q, K.transpose(-2, -1)) # similarity matrix bxnxn
        attention = attention/math.sqrt(C)
        attention = torch.nn.functional.softmax(attention, dim=1)
        attention = torch.matmul(attention, V) # bxnxh
        y = attention

        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_size, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class Model(nn.Module):
    def __init__(self, hidden_dim=256,
        hidden_layers=4,
        rmsnorm=False,
        activation='gelu',
        vocab_size=10000, block_size, n_layer, n_embd, n_head, 
        embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, self.hidden_dim),
            wpe = nn.Embedding(self.block_size, self.hidden_dim),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(self.hidden_dim),
        ))
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

     def forward(self, idx, targets=None, mask=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

         # forward pass
        tok_emb = self.transformer.wte(idx) # token embeddings
        pos_emb = self.transformer.wpe(pos) # position embeddings
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # calculate loss if targets are provided
        loss = None
        if targets is not None:
            logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets = targets[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(logits, targets, reduction='none')
            if mask is not None:
                loss = loss.view(b, -1)
                mask = mask[:, 1:].contiguous()
                assert 0 not in mask.sum(dim=1), "each sequence must have at least one token unmasked"
                loss = loss * mask
                loss = loss.sum() / mask.sum()
            else:
                loss = loss.mean()

        return logits, loss