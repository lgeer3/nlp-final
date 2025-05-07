import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from types import SimpleNamespace
from transformers import PretrainedConfig
from transformers import PreTrainedModel

class CustomGPTConfig(PretrainedConfig):
    model_type = "custom_gpt"

    def __init__(self, 
                 hidden_dim=256,
                 hidden_layers=6,
                 vocab_size=50257,
                 block_size=1024,
                 n_head=8,
                 attn_pdrop=0.2,
                 resid_pdrop=0.2,
                 embd_pdrop=0.2,
                 norm_type="layernorm",
                 activation="gelu",
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.norm_type = norm_type
        self.activation = activation


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class GeGLU(nn.Module):
    def __init__(self, hidden_dim, inner_dim=None):
        super().__init__()
        inner_dim = inner_dim or hidden_dim * 4
        self.proj = nn.Linear(hidden_dim, 2 * inner_dim)
        self.gelu = NewGELU()
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.gelu(gate)
         
 
 
class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, inner_dim = None):
        super().__init__()
        inner_dim = inner_dim or hidden_dim * 4
        self.proj = nn.Linear(hidden_dim, 2 * inner_dim)
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.sigmoid(gate) # SiLU
 
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
 
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
 
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
 
class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_type="layernorm"):
        super().__init__()
        self.fn = fn
        if norm_type == "layernorm":
            self.norm = nn.LayerNorm(dim)
        elif norm_type == "rmsnorm":
            self.norm = RMSNorm(dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
 
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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
        self.dropout = attn_pdrop
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size)).bool())

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.hidden_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)


        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            mask = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            att = att.masked_fill(mask == 0, float("-inf"))
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

class Model(PreTrainedModel):
    """ The full GPT language model with a context size of block_size """
    def __init__(self, config: CustomGPTConfig):
        super().__init__(config)
        self.config = config
        self.hidden_dim = self.config.hidden_dim
        self.vocab_size = self.config.vocab_size
        self.block_size = self.config.block_size

        # input embedding stem
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.hidden_dim)
        self.pos_emb = nn.Embedding(self.config.block_size, self.config.hidden_dim)
        self.drop = nn.Dropout(self.config.embd_pdrop)

        # transformer
        self.blocks = nn.ModuleList()
        for _ in range(self.config.hidden_layers):
            # Attention part with PreNorm
            attn = PreNorm(
                self.config.hidden_dim,
                CausalSelfAttention(self.config.hidden_dim, self.config.n_head, self.config.block_size, self.config.attn_pdrop, self.config.resid_pdrop),
                norm_type=self.config.norm_type
            )
             
            # MLP part
            if self.config.activation == "gelu":
                mlp = nn.Sequential(
                    nn.Linear(self.config.hidden_dim, 4 * self.config.hidden_dim),
                    NewGELU(),
                    nn.Linear(4 * self.config.hidden_dim, self.config.hidden_dim),
                    nn.Dropout(self.config.resid_pdrop),
                )
            elif self.config.activation == "geglu":
                mlp = nn.Sequential(
                    GeGLU(self.config.hidden_dim),
                    nn.Linear(4 * self.config.hidden_dim, self.config.hidden_dim),
                    nn.Dropout(self.config.resid_pdrop),
                )
            elif self.config.activation == "swiglu":
                mlp = nn.Sequential(
                    SwiGLU(self.config.hidden_dim),
                    nn.Linear(4 * self.config.hidden_dim, self.config.hidden_dim),
                    nn.Dropout(self.config.resid_pdrop),
                )
            else:
                raise ValueError(f"Unknown mlp_type: {self.config.activation}")
             
            # MLP with PreNorm
            mlp = PreNorm(self.config.hidden_dim, mlp, norm_type=self.config.norm_type)
             
            # Combine into a block
            block = nn.ModuleDict({
                'attn': attn,
                'mlp': mlp,
            })
            self.blocks.append(block)
 
        
        # decoder head
        self.ln_f = RMSNorm(self.config.hidden_dim) if self.config.norm_type == "rmsnorm" else nn.LayerNorm(self.config.hidden_dim)
        self.head = nn.Linear(self.config.hidden_dim, self.config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)

        self.token2id = self.config.token2id
        self.banned_token_ids = []
        if self.config.token2id is not None:
            self.banned_token_ids = [id for tok, id in self.config.token2id.items() if tok.startswith("[unused")]



    '''
    def __init__(self, hidden_dim=256, hidden_layers=6, vocab_size=50257, 
                 block_size=1024, n_head=8, attn_pdrop=0.2, resid_pdrop=0.2, 
                 embd_pdrop=0.2, token2id: Optional[dict] = None,
                 norm_type = "layernorm", 
                 activation = "gelu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.block_size = block_size

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(block_size, hidden_dim)
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.ModuleList()
        for _ in range(hidden_layers):
            # Attention part with PreNorm
            attn = PreNorm(
                hidden_dim,
                CausalSelfAttention(hidden_dim, n_head, block_size, attn_pdrop, resid_pdrop),
                norm_type=norm_type
            )
             
            # MLP part
            if activation == "gelu":
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, 4 * hidden_dim),
                    NewGELU(),
                    nn.Linear(4 * hidden_dim, hidden_dim),
                    nn.Dropout(resid_pdrop),
                )
            elif activation == "geglu":
                mlp = nn.Sequential(
                    GeGLU(hidden_dim),
                    nn.Linear(4 * hidden_dim, hidden_dim),
                    nn.Dropout(resid_pdrop),
                )
            elif activation == "swiglu":
                mlp = nn.Sequential(
                    SwiGLU(hidden_dim),
                    nn.Linear(4 * hidden_dim, hidden_dim),
                    nn.Dropout(resid_pdrop),
                )
            else:
                raise ValueError(f"Unknown mlp_type: {activation}")
             
            # MLP with PreNorm
            mlp = PreNorm(hidden_dim, mlp, norm_type=norm_type)
             
            # Combine into a block
            block = nn.ModuleDict({
                'attn': attn,
                'mlp': mlp,
            })
            self.blocks.append(block)
 
        
        # decoder head
        self.ln_f = RMSNorm(hidden_dim) if norm_type == "rmsnorm" else nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)

        self.token2id = token2id
        self.banned_token_ids = []
        if token2id is not None:
            self.banned_token_ids = [id for tok, id in token2id.items() if tok.startswith("[unused")]
    '''


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            torch.nn.init.zeros_(module.bias) if hasattr(module, 'bias') else None
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None, mask=None):
        device = idx.device
        b, t = idx.size()
        
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        pos = torch.arange(0, t, dtype=torch.long, device=device) # each position maps to a (learnable) vector
        position_embeddings = self.pos_emb(pos)

        x = self.drop(token_embeddings + position_embeddings)
        for block in self.blocks:
            x = x + block['attn'](x)
            x = x + block['mlp'](x)
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
        

        return SimpleNamespace(logits=logits, loss=loss)

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b, t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond).logits
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if hasattr(self, "banned_token_ids"):
                logits[:, self.banned_token_ids] = -float("inf")

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