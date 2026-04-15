import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x):
        B, T, C = x.shape  # batch, tokens, channels

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # split heads
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # attention
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # causal mask (block future tokens)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)  # (T, T)
        scores = scores.masked_fill(mask == 1, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = weights @ V

        # combine heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.W_o(out)
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff=4*d_model) # Expanded for better capacity
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-LayerNorm Residual Connections
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
    

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

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]


class TinyStoriesLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, max_seq_len=1000, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, y=None):
        x = self.embedding(x)
        x = self.pos_embedding(x)
        x = self.transformer_blocks(x)
        x = self.lm_head(x)

        if y is not None:
            # Compute loss
            B, T, C = x.shape
            x = x.view(B * T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(x, y)
            return x, loss

        return x, None
    

    def generate(
        self,
        idx,
        tokenizer,
        max_new_tokens=100,
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
            # Focus only on the last token's logits: (B, C)
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
            for _ in range(max_new_tokens):
                # Crop context to max_len so positional embeddings don't overflow
                idx_cond = idx if idx.size(1) <= self.pos_embedding.pe.size(1) else idx[:, -self.pos_embedding.pe.size(1):]
                
                # Get predictions
                logits, _ = self(idx_cond)
                
                # Sample next token
                next_token = sample(logits)
                
                if next_token.item() == eos_token_id:
                    break
                
                idx = torch.cat((idx, next_token), dim=1)
                yield tokenizer.decode([next_token.item()])

        if stream:
            return _generate_tokens(idx)
        else:
            generated_tokens = []
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    idx_cond = idx if idx.size(1) <= self.pos_embedding.pe.size(1) else idx[:, -self.pos_embedding.pe.size(1):]
                    logits, _ = self(idx_cond)
                    next_token = sample(logits)
                    
                    if next_token.item() == eos_token_id:
                        break
                        
                    idx = torch.cat((idx, next_token), dim=1)
                    generated_tokens.append(next_token.item())
            
            return tokenizer.decode(generated_tokens)