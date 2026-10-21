"""
Mechanistic Interpretability from Scratch: Find the Induction Circuit in a Tiny GPT

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_batch
import torch

def make_batch(B, gen, V=32, T=48, nmin=6, nmax=24):
    # Generate random token IDs for every sequence.
    tokens = torch.randint(0, V, (B, T), generator=gen)

    # Sample a repeat length independently for each sequence.
    # The upper bound is inclusive because torch.randint excludes its upper bound.
    n = torch.randint(nmin, nmax + 1, (B,), generator=gen)

    # Copy the first n[i] tokens into the next n[i] positions in each row.
    for i in range(B):
        repeat_len = int(n[i].item())
        tokens[i, repeat_len:2 * repeat_len] = tokens[i, :repeat_len].clone()

    # Return the generated sequences and their individual repeat lengths.
    return tokens, n

# Step 2 - Head
import math
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, d, d_head):
        super().__init__()

        # Store the attention head dimension.
        self.d_head = d_head

        # Query, key, and value projection matrices: d -> d_head.
        self.W_Q = nn.Linear(d, d_head, bias=False)
        self.W_K = nn.Linear(d, d_head, bias=False)
        self.W_V = nn.Linear(d, d_head, bias=False)

        # Output projection matrix: d_head -> d.
        self.W_O = nn.Linear(d_head, d, bias=False)

    def forward(self, x, mask, x_k=None, x_v=None):
        # Queries are always calculated from the primary residual stream.
        q = self.W_Q(x)

        # Use separate residual streams for keys and values when provided.
        # Otherwise, use x for both.
        key_input = x if x_k is None else x_k
        value_input = x if x_v is None else x_v

        k = self.W_K(key_input)
        v = self.W_V(value_input)

        # Shape transformations:
        # q: (B, T, d_head)
        # k: (B, d_head, T)
        # v: (B, T, d_head)
        k = k.transpose(1, 2)

        # Scaled dot-product attention scores: (B, T, T).
        scores = (q @ k) / math.sqrt(self.d_head)

        # The boolean mask marks forbidden attention positions with True.
        # Broadcast the (T, T) mask across the batch dimension.
        scores = scores.masked_fill(mask, -1e9)

        # Normalize over the key-position dimension.
        # Attention pattern shape: (B, T, T).
        a = scores.softmax(dim=-1)

        # Weighted sum of value vectors: (B, T, d_head).
        head_output = a @ v

        # Project the head output back into the residual stream dimension.
        out = self.W_O(head_output)

        # Return both the projected output and attention pattern.
        return out, a

