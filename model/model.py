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

    def __init__(self, hidden_dim, n_head, block_size, attn_pdrop, resid_pdrop):
        super().__init__()
        assert hidden_dim % n_head == 0
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_head
        self.block_size = block_size

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(hidden_dim, 3 * hidden_dim)
        # output projection
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.hidden_dim, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ A standard transformer block with layer norm and residual connections """

    def __init__(self, hidden_dim, n_head, block_size, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, n_head, block_size, attn_pdrop, resid_pdrop)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            NewGELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        # Attention part
        x = x + self.attn(self.ln_1(x))
        # MLP part
        x = x + self.mlp(self.ln_2(x))
        return x

class Model(nn.Module):
    """ The full GPT language model with a context size of block_size """

    def __init__(self, hidden_dim=256, hidden_layers=6, vocab_size=50257, 
                 block_size=1024, n_head=8, attn_pdrop=0.1, resid_pdrop=0.1, 
                 embd_pdrop=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.block_size = block_size

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, hidden_dim))
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[
            Block(hidden_dim, n_head, block_size, attn_pdrop, resid_pdrop)
            for _ in range(hidden_layers)
        ])
        
        # decoder head
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # init all weights
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
        
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # shift logits and targets for next-token prediction
            logits = logits[:, :-1, :].contiguous()
            targets = targets[:, 1:].contiguous()
            
            # flatten the tokens
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                reduction='none'
            )
            
            # apply mask if provided
            if mask is not None:
                mask = mask[:, 1:].contiguous()
                loss = loss.view(b, -1)
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = loss.mean()

        return {'logits': logits, 'loss': loss}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b, t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)["logits"]
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx