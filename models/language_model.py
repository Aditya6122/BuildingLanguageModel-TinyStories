"""
Language model implementation using GRU-like architecture.

This module defines the LanguageModel class with gated recurrent units,
layer normalization, and text generation capabilities.
"""

import torch
import torch.nn as nn
from torch.functional import F


class LanguageModel(nn.Module):
    """
    A language model based on GRU architecture with layer normalization.

    This model uses reset, update, and candidate gates similar to GRU,
    but with layer normalization applied to stabilize training.
    """

    def __init__(self, vocab_size, embed_dim=512, hidden_dim=1024):
        """
        Initialize the language model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of the embedding layer.
            hidden_dim (int): Dimension of the hidden state.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Embedding Layer for input text.

        self.Wr = nn.Linear(embed_dim + hidden_dim, hidden_dim)  # Reset Gate Weights: Decides what to ignore from past for the output.
        self.Wu = nn.Linear(embed_dim + hidden_dim, hidden_dim)  # Update/Filter Gate: Decides what to keep and remove from the past memory.
        self.Wc = nn.Linear(embed_dim + hidden_dim, hidden_dim)  # Candidate/Proposal Gate: Proposes what to add in the memory.

        self.Wo = nn.Linear(hidden_dim, vocab_size)  # Output Projection Weights: Decides what to project/show in the output.

        self.ln_combined = nn.LayerNorm(embed_dim + hidden_dim)  # Layer Normalization for concatenated features.
        self.ln_hidden = nn.LayerNorm(hidden_dim)  # Layer Normalization for hidden features.

    def step(self, x_t, memory):
        """
        Perform one step of the recurrent computation.

        Args:
            x_t (torch.Tensor): Input token at current timestep, shape (B,).
            memory (torch.Tensor): Previous hidden state, shape (B, H).

        Returns:
            tuple: (logits, new_memory) where logits shape (B, V) and memory shape (B, H).
        """
        x_emb = self.embedding(x_t)  # (B, D)
        combined = torch.cat([memory, x_emb], dim=1)  # (B, H+D)
        combined = self.ln_combined(combined)

        reset_gate_activations = torch.sigmoid(self.Wr(combined))  # (B, H)
        reset_memory = reset_gate_activations * memory  # (B, H)

        reset_combined = torch.cat([reset_memory, x_emb], dim=1)  # (B, H+D)
        reset_combined = self.ln_combined(reset_combined)

        candidate_memory = torch.tanh(self.Wc(reset_combined))  # (B, H)  # Fixed: use Wc for candidate
        candidate_memory = self.ln_hidden(candidate_memory)

        update_gate_activations = torch.sigmoid(self.Wu(combined))  # (B, H)  # Fixed: use Wu for update

        final_memory = update_gate_activations * memory + (1 - update_gate_activations) * candidate_memory  # (B, H)
        logits = self.Wo(final_memory)  # (B, V)

        return logits, final_memory

    def forward(self, x, y=None):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input sequence, shape (B, T).
            y (torch.Tensor, optional): Target sequence, shape (B, T).

        Returns:
            tuple: (logits, memory, loss) where loss is None if y is None.
        """
        B, T = x.shape
        device = x.device

        memory = torch.zeros(B, self.hidden_dim, device=device)  # Initialize memory with zeros
        logits_seq = []

        for t in range(T):
            logits, memory = self.step(x[:, t], memory)
            logits_seq.append(logits)

        logits = torch.stack(logits_seq, dim=1)  # (B, T, V)

        if y is None:
            return logits, memory, None

        loss = F.cross_entropy(
            logits.view(B * T, -1),
            y.view(B * T)
        )

        return logits, memory, loss

    def generate(self, x, max_new_tokens=50):
        """
        Generate text autoregressively.

        Args:
            x (torch.Tensor): Initial prompt, shape (B, T).
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            torch.Tensor: Generated tokens, shape (B, max_new_tokens).
        """
        self.eval()
        B = x.shape[0]
        device = x.device

        memory = torch.zeros(B, self.hidden_dim, device=device)

        with torch.no_grad():
            for t in range(x.shape[1]):
                _, memory = self.step(x[:, t], memory)

            last_token = x[:, -1]  # start from last token
            generated = []

            for _ in range(max_new_tokens):
                logits, memory = self.step(last_token, memory)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(1)

                generated.append(next_token)
                last_token = next_token

        return torch.stack(generated, dim=1)