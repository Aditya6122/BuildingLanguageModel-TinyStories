import math
from matplotlib.pylab import block
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model # Standard expansion

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(), 
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, pad_mask=None, past_k=None, past_v=None):
        B, T, C = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # split heads
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # 🔥 KV CACHE LOGIC
        if past_k is not None and past_v is not None:
            # concatenate past and current
            K = torch.cat([past_k, K], dim=2)  # (B, heads, T_total, d_k)
            V = torch.cat([past_v, V], dim=2)

        # save new cache
        new_k, new_v = K, V

        # attention
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if past_k is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
            scores = scores.masked_fill(mask == 1, float('-inf'))

        if pad_mask is not None and past_k is None:
            scores = scores.masked_fill(pad_mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = weights @ V

        # combine heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.W_o(out), new_k, new_v

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff=4*d_model) # Expanded for better capacity
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None, past_k=None, past_v=None):
        attn_out, new_k, new_v = self.attn(
            self.ln1(x),
            pad_mask=pad_mask,
            past_k=past_k,
            past_v=past_v
        )

        x = x + attn_out
        x = x + self.ff(self.ln2(x))

        return x, new_k, new_v  
    

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()

        # Create positional encoding matrix (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) 

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)        
        self.register_buffer("pe", pe) # Register as buffer (not trainable, but moves with model)

    def forward(self, x, start_pos=0):
        """
        x: (batch_size, seq_len, d_model)
        start_pos: where this chunk starts in sequence
        """
        return x + self.pe[:, start_pos:start_pos + x.size(1)]


class TinyStoriesLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, max_seq_len=1000, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, y=None, pad_token_id=None, past_kvs=None):
        pad_mask = (x != pad_token_id).unsqueeze(1).unsqueeze(2) if pad_token_id is not None else None

        # 🔥 compute starting position
        if past_kvs is None:
            start_pos = 0
        else:
            # length of cached sequence
            start_pos = past_kvs[0][0].size(2)

        x = self.embedding(x)
        x = self.pos_embedding(x, start_pos=start_pos)

        new_kvs = []

        for i, block in enumerate(self.transformer_blocks):
            past_k, past_v = (past_kvs[i] if past_kvs is not None else (None, None))

            x, new_k, new_v = block(x, pad_mask=pad_mask, past_k=past_k, past_v=past_v)
            new_kvs.append((new_k, new_v))

        x = self.lm_head(x)

        if y is not None:
            B, T, C = x.shape
            x = x.view(B * T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(x, y, ignore_index=0)
            return x, loss

        return x, new_kvs
    

    def generate(
        self,
        idx,
        tokenizer,
        max_new_tokens=500,
        temperature=1.0,
        top_k=None,
        top_p=None,
        stream=False,
    ):
        self.eval()
        device = idx.device
        eos_token_id = tokenizer.token_to_id("<|end_of_text|>")
        temperature = max(temperature, 1e-5)

        def sample(logits):
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        def _generate_tokens(idx):
            past_kvs = None  # 🔥 KV cache initialization

            for step in range(max_new_tokens):

                # 🔥 First step: full sequence
                # 🔥 Later steps: only last token
                if past_kvs is None:
                    idx_cond = idx
                else:
                    idx_cond = idx[:, -1:]

                logger.debug(f"Step {step} | Input shape: {idx_cond.shape}")

                # 🔥 Forward with KV cache
                logits, past_kvs = self(idx_cond, past_kvs=past_kvs)

                # Sample next token
                next_token = sample(logits)

                if next_token.item() == eos_token_id:
                    break

                # Append to sequence
                idx = torch.cat((idx, next_token), dim=1)

                output_token = tokenizer.decode([next_token.item()])
                yield output_token

        if stream:
            return _generate_tokens(idx)

        else:
            generated_tokens = []
            past_kvs = None  # 🔥 KV cache

            with torch.no_grad():
                for step in range(max_new_tokens):

                    # 🔥 Same logic as above
                    if past_kvs is None:
                        idx_cond = idx
                    else:
                        idx_cond = idx[:, -1:]

                    logits, past_kvs = self(idx_cond, past_kvs=past_kvs)

                    next_token = sample(logits)

                    if next_token.item() == eos_token_id:
                        break

                    idx = torch.cat((idx, next_token), dim=1)
                    generated_tokens.append(next_token.item())

            return tokenizer.decode(generated_tokens)