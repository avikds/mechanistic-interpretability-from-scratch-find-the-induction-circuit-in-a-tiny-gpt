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

# Step 3 - AttnOnlyGPT
class AttnOnlyGPT(nn.Module):
    def __init__(self, V=32, T=48, d=64, heads=2, d_head=32, layers=2):
        super().__init__()

        # Token and positional embeddings.
        self.wte = nn.Embedding(V, d)
        self.wpe = nn.Embedding(T, d)

        # Each layer contains independent attention heads.
        # Every head in a layer receives the same normalized residual stream.
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Head(d, d_head) for _ in range(heads)
            ])
            for _ in range(layers)
        ])

        # Pre-layer normalization for each attention layer.
        self.lns = nn.ModuleList([
            nn.LayerNorm(d) for _ in range(layers)
        ])

        # Final layer normalization and vocabulary projection.
        self.lnf = nn.LayerNorm(d)
        self.unembed = nn.Linear(d, V, bias=False)

    def forward(self, idx, cache=None, patch=None):
        # idx has shape (B, T'), where B is batch size and T' is sequence length.
        B, T_seq = idx.shape

        # Add token embeddings and positional embeddings.
        positions = torch.arange(T_seq, device=idx.device)
        x = self.wte(idx) + self.wpe(positions)

        # Save the initial residual stream for mechanistic analysis.
        if isinstance(cache, dict):
            cache["embed"] = x.detach()

        # True indicates a forbidden attention position.
        # Each token can attend only to itself and preceding tokens.
        mask = torch.triu(
            torch.ones(
                T_seq,
                T_seq,
                dtype=torch.bool,
                device=idx.device
            ),
            diagonal=1
        )

        # Process each transformer layer.
        for l, layer in enumerate(self.layers):
            # Record the residual stream entering this layer, before LayerNorm.
            if isinstance(cache, dict):
                cache[("resid_pre", l)] = x.detach()

            # Pre-LayerNorm: all heads read the same normalized input.
            inp = self.lns[l](x)

            # Sum the outputs of independent attention heads.
            head_sum = torch.zeros_like(x)

            for h, head in enumerate(layer):
                # Compute the head's output and attention pattern.
                head_out, pattern = head(inp, mask)

                head_key = ("head", l, h)
                pattern_key = ("pattern", l, h)

                # Optionally replace this head's output with a patched tensor.
                # The patch must have shape (B, T', d).
                if isinstance(patch, dict) and head_key in patch:
                    head_out = patch[head_key]

                # Cache the actual head output, including any replacement.
                if isinstance(cache, dict):
                    cache[head_key] = head_out.detach()
                    cache[pattern_key] = pattern.detach()

                head_sum = head_sum + head_out

            # Residual connection: add the combined outputs to the stream.
            x = x + head_sum

        # Cache the final residual stream before final normalization.
        if isinstance(cache, dict):
            cache["resid_final"] = x.detach()

        # Convert the normalized residual stream into vocabulary logits.
        logits = self.unembed(self.lnf(x))

        return logits

