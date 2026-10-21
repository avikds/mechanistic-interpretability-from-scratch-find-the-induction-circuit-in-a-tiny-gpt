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

