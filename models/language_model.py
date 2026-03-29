"""
Language model implementation using GRU-like architecture.

This module defines the LanguageModel class with gated recurrent units,
layer normalization, and text generation capabilities.

The model processes sequences token-by-token using GRU cells that maintain
a hidden state (memory) across timesteps, with gating mechanisms to control
information flow and prevent vanishing gradients.
"""

import torch
import torch.nn as nn
from torch.functional import F


class LanguageModel(nn.Module):
    """
    A language model based on GRU (Gated Recurrent Unit) architecture with layer normalization.

    This model implements a simplified GRU with:
    - Reset gate: Controls what information from previous state to forget
    - Update gate: Controls how much to update the hidden state
    - Candidate gate: Computes new memory candidate
    - Layer normalization for training stability
    - Output projection to vocabulary logits

    Architecture: Input → Embedding → GRU steps → Output projection → Logits
    """

    def __init__(self, vocab_size, embedding_dimension=512, hidden_dimension=1024):
        """
        Initialize the language model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of the embedding layer.
            hidden_dim (int): Dimension of the hidden state.
        """
        super().__init__()
        self.hidden_dim = hidden_dimension
        self.embed_dim = embedding_dimension
        self.hidden_dim = hidden_dimension
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # Embedding Layer for input text.

        # GRU Gate weights:
        # W_z: Update gate - controls how much of previous memory to keep
        # W_r: Reset gate - controls how much of previous memory to forget
        # W_n: Candidate gate - computes new memory candidate
        # W_out: Output projection - maps hidden state to vocabulary logits
        self.W_z = nn.Linear(self.embed_dim + self.hidden_dim, self.hidden_dim)
        self.W_r = nn.Linear(self.embed_dim + self.hidden_dim, self.hidden_dim)
        self.W_n = nn.Linear(self.embed_dim + self.hidden_dim, self.hidden_dim)
        self.W_out = nn.Linear(self.hidden_dim, self.vocab_size)

        # Layer normalization for stability
        self.ln_combined = nn.LayerNorm(self.embed_dim + self.hidden_dim)
        self.ln_hidden = nn.LayerNorm(self.hidden_dim)

    def step(self, x_t, memory):
        """
        Single GRU step: compute next hidden state and logits from input token and previous memory.

        Args:
            x_t: Input token indices (B,)
            memory: Previous hidden state (B, H)

        Returns:
            logits: Output logits for next token prediction (B, V)
            final_memory: Updated hidden state (B, H)
        """
        # Embed input token
        x_emb = self.embedding(x_t)                  # (B, D)

        # Concatenate previous memory and current input embedding
        combined = torch.cat([memory, x_emb], dim=1) # (B, H+D)
        combined = self.ln_combined(combined)        # Layer norm for stability

        # Reset gate: decides what parts of previous memory to forget
        reset_gate_activations = torch.sigmoid(self.W_r(combined)) # (B, H)
        reset_memory = reset_gate_activations * memory # (B, H) - selectively forget

        # Compute candidate memory with reset memory
        reset_combined = torch.cat([reset_memory, x_emb], dim=1) # (B, H+D)
        reset_combined = self.ln_combined(reset_combined)
        candidate_memory = torch.tanh(self.W_n(reset_combined)) # (B, H) - new candidate
        candidate_memory = self.ln_hidden(candidate_memory)

        # Update gate: decides how much to update from candidate vs keep old
        update_gate_activations = torch.sigmoid(self.W_z(combined))  # (B, H)

        # Final memory: interpolation between old memory and candidate
        final_memory = update_gate_activations * memory + (1 - update_gate_activations) * candidate_memory # (B, H)

        # Output logits for next token prediction
        logits = self.W_out(final_memory)  # (B, V)

        return logits, final_memory

    def forward(self, x, y=None):
        """
        Process a sequence of tokens through the GRU language model.

        Args:
            x: Input token sequences (B, T) - batch_size x sequence_length
            y: Target token sequences (B, T) - for training loss computation

        Returns:
            logits: Token prediction logits (B, T, V)
            memory: Final hidden state (B, H)
            loss: Cross-entropy loss if y provided, else None
        """
        B, T = x.shape
        device = x.device

        # Initialize hidden state (memory) to zeros
        memory = torch.zeros(B, self.hidden_dim, device=device)
        logits_seq = []

        # Process each token in sequence
        for t in range(T):
            logits, memory = self.step(x[:, t], memory)  # Step through one token
            logits_seq.append(logits)

        # Stack logits for all timesteps: (B, T, V)
        logits = torch.stack(logits_seq, dim=1)

        if y is None:
            return logits, memory, None

        # Compute cross-entropy loss if targets provided
        loss = F.cross_entropy(
            logits.view(B*T, -1),  # Flatten to (B*T, V)
            y.view(B*T)            # Flatten to (B*T,)
        )

        return logits, memory, loss

    def generate(
        self,
        x,
        tokenizer,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        top_p=None,
        stream=False,
    ):
        """
        Generate text continuation from input sequence using the language model.

        Args:
            x: Input token sequence (B, T) or text string
            tokenizer: Tokenizer for encoding/decoding
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            stream: Whether to yield tokens one by one

        Returns:
            Generated text string, or token-by-token generator if stream=True
        """
        self.eval()
        B = x.shape[0]
        device = x.device

        memory = torch.zeros(B, self.hidden_dim, device=device)
        eos_token_id = tokenizer.token_to_id("<|end_of_text|>")

        temperature = max(temperature, 1e-5)

        def sample(logits):
            if temperature == 0:
                return torch.argmax(logits, dim=-1)

            logits_ = logits / temperature

            if top_k is not None:
                k = min(top_k, logits_.size(-1))
                values, indices = torch.topk(logits_, k)
                probs = F.softmax(values, dim=-1)
                return indices.gather(-1, torch.multinomial(probs, 1)).squeeze(1)

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits_, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)

                mask = cumulative_probs > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False

                sorted_logits[mask] = -float("inf")
                probs = F.softmax(sorted_logits, dim=-1)

                return sorted_indices.gather(
                    -1, torch.multinomial(probs, 1)
                ).squeeze(1)

            probs = F.softmax(logits_, dim=-1)
            return torch.multinomial(probs, 1).squeeze(1)

        def _generate_tokens():
            """Generator function for streaming output."""
            nonlocal memory
            with torch.no_grad():
                # Process prompt
                for t in range(x.shape[1]):
                    _, memory = self.step(x[:, t], memory)

                last_token = x[:, -1]

                for _ in range(max_new_tokens):
                    logits, memory = self.step(last_token, memory)
                    next_token = sample(logits)
                    token_id = next_token.item()

                    if token_id == eos_token_id:
                        break

                    # Decode the new token with proper whitespace handling
                    token_text = tokenizer.decode([token_id])
                    yield token_text

                    last_token = next_token

        if stream:
            return _generate_tokens()
        else:
            generated_tokens = []
            with torch.no_grad():
                # Process prompt
                for t in range(x.shape[1]):
                    _, memory = self.step(x[:, t], memory)

                last_token = x[:, -1]

                for _ in range(max_new_tokens):
                    logits, memory = self.step(last_token, memory)
                    next_token = sample(logits)
                    token_id = next_token.item()
                    generated_tokens.append(token_id)

                    if token_id == eos_token_id:
                        break

                    last_token = next_token
                
            generated_text = tokenizer.decode(generated_tokens)

            return generated_text