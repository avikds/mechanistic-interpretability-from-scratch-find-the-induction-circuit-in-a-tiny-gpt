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

# Step 4 - icl_accuracy
@torch.no_grad()
def icl_accuracy(model, tokens, n):
    # Predict the next token at each position except the final input position.
    logits = model(tokens[:, :-1])
    predictions = logits.argmax(dim=-1)

    hits = 0
    total = 0

    # Score only the second copy, excluding its final token position.
    for i in range(tokens.size(0)):
        repeat_len = int(n[i].item())

        # Score prediction positions [n, 2*n - 1).
        # This gives exactly n - 1 next-token predictions.
        start = repeat_len
        end = 2 * repeat_len - 1

        # Prediction at position t must match the token at position t + 1.
        predicted = predictions[i, start:end]
        targets = tokens[i, start + 1:end + 1]

        hits += (predicted == targets).sum().item()
        total += end - start

    # Return the accuracy as a Python float.
    return float(hits / total) if total > 0 else 0.0

# Step 5 - train_model
import torch.nn.functional as F

def train_model(model, steps, seed, lr=1e-3, batch=64, log_every=100, probe=None):
    # Read vocabulary size and context length from the model.
    V = model.wte.weight.shape[0]
    T = model.wpe.weight.shape[0]

    # Restrict repeat lengths to what the model's context can accommodate.
    nmax = T // 2
    nmin = min(6, nmax)

    # Use a dedicated generator for reproducible training batches.
    gen = torch.Generator().manual_seed(seed)

    # Prepare Adam optimizer.
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Create a fixed evaluation batch for comparable ICL accuracy scores.
    eval_gen = torch.Generator().manual_seed(999)
    eval_tokens, eval_n = make_batch(
        256, eval_gen, V, T, nmin, nmax
    )

    # Store training and evaluation measurements.
    history = []

    for s in range(steps):
        # Generate a batch with the model-compatible dimensions.
        tokens, n = make_batch(batch, gen, V, T, nmin, nmax)

        # Predict each next token from the preceding tokens.
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]

        # Compute cross-entropy across all batch and sequence positions.
        loss = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1)
        )

        # Perform one optimization step.
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Record metrics at the requested interval and on the final step.
        if s % log_every == 0 or s == steps - 1:
            record = {
                "step": s,
                "loss": float(loss.item()),
                "icl_acc": float(icl_accuracy(model, eval_tokens, eval_n)),
            }

            # Include any additional circuit or diagnostic measurements.
            if probe is not None:
                record.update(probe(model))

            history.append(record)

    return history

# Step 6 - induction_score
def induction_score(A, n):
    # A has shape (B, T, T); n contains one repeat length per row.
    B, T, _ = A.shape

    scores = []

    for i in range(B):
        repeat_len = int(n[i].item())

        # Consider query positions in [n[i], 2 * n[i] - 1).
        # Each query attends to the token following the first occurrence
        # of its current token.
        for t in range(repeat_len, 2 * repeat_len - 1):
            key_pos = t - repeat_len + 1
            scores.append(A[i, t, key_pos])

    # Average all selected attention scores across rows and positions.
    if not scores:
        return 0.0

    return float(torch.stack(scores).mean().item())


def prev_token_score(A):
    # A has shape (B, T, T).
    _, T, _ = A.shape

    # Handle sequences without any preceding-token positions.
    if T <= 1:
        return 0.0

    # Extract the first sub-diagonal: attention from position t to t - 1.
    positions = torch.arange(1, T, device=A.device)
    scores = A[:, positions, positions - 1]

    # Average over every batch row and every query position t >= 1.
    return float(scores.mean().item())

# Step 7 - head_scores
def head_scores(model, seed=999, B=64):
    # Read the vocabulary size and context length from the model.
    V = model.wte.weight.shape[0]
    T = model.wpe.weight.shape[0]

    # Use repeat lengths compatible with the model's context.
    nmax = T // 2
    nmin = min(6, nmax)

    # Generate a reproducible batch of repeated-token sequences.
    gen = torch.Generator().manual_seed(seed)
    tokens, n = make_batch(B, gen, V, T, nmin, nmax)

    # Capture the attention patterns from a single forward pass.
    cache = {}
    with torch.no_grad():
        model(tokens, cache=cache)

    # Calculate both scores for every attention head.
    scores = {}

    for l, layer in enumerate(model.layers):
        for h, _ in enumerate(layer):
            A = cache[("pattern", l, h)]

            scores[(l, h)] = {
                "induction": induction_score(A, n),
                "prev_token": prev_token_score(A),
            }

    return scores


def classify_heads(scores, ind_thresh=0.5, prev_thresh=0.25):
    # Store the identified heads in separate, mutually exclusive categories.
    result = {
        "induction": [],
        "previous": [],
    }

    for (l, h), score in scores.items():
        if score["induction"] >= ind_thresh:
            # Give induction classification priority if both thresholds pass.
            result["induction"].append((l, h))
        elif score["prev_token"] >= prev_thresh:
            result["previous"].append((l, h))

    # Return deterministic, sorted lists of (layer, head) pairs.
    result["induction"].sort()
    result["previous"].sort()

    return result

# Step 8 - phase_change
def score_probe(model):
    # Measure attention-pattern scores for every head.
    scores = head_scores(model)

    # Find the largest induction and previous-token scores across all heads.
    max_induction = max(
        (score["induction"] for score in scores.values()),
        default=0.0
    )

    max_prev = max(
        (score["prev_token"] for score in scores.values()),
        default=0.0
    )

    # Return scalar metrics suitable for train_model's probe argument.
    return {
        "max_induction": float(max_induction),
        "max_prev": float(max_prev),
    }


def phase_change(history, acc_thresh=0.5, score_thresh=0.5):
    # Find the first step at which in-context accuracy reaches its threshold.
    acc_step = next(
        (
            record["step"]
            for record in history
            if record["icl_acc"] >= acc_thresh
        ),
        None
    )

    # Find the first step at which the maximum induction score reaches
    # its threshold.
    induction_step = next(
        (
            record["step"]
            for record in history
            if record["max_induction"] >= score_thresh
        ),
        None
    )

    # Handle an empty history without raising an IndexError.
    if not history:
        final_acc = None
        final_induction = None
    else:
        # Report the metrics recorded at the final training step.
        final_acc = history[-1]["icl_acc"]
        final_induction = history[-1]["max_induction"]

    return {
        "acc_step": acc_step,
        "induction_step": induction_step,
        "final_acc": final_acc,
        "final_induction": final_induction,
    }

# Step 9 - qk_circuit
def qk_circuit(head):
    # W_Q and W_K have shape (d_head, d).
    # The QK circuit maps residual-stream vectors to attention scores:
    # x_i^T W_Q^T W_K x_j.
    W_Q = head.W_Q.weight
    W_K = head.W_K.weight

    # Result shape: (d, d).
    # Detach the result because this function analyzes weights, not gradients.
    return (W_Q.T @ W_K).detach()


def ov_circuit(head):
    # W_V has shape (d_head, d), and W_O has shape (d, d_head).
    # The OV circuit maps an attended residual-stream vector through
    # the value projection and output projection.
    W_V = head.W_V.weight
    W_O = head.W_O.weight

    # Result shape: (d, d).
    return (W_O @ W_V).detach()

# Step 10 - copying_score
def copying_score(OV):
    # Compute the eigenvalues of the OV circuit.
    # Eigenvalues may be complex, so retain their real parts.
    eigvals = torch.linalg.eigvals(OV)
    real_parts = eigvals.real

    # Measure the fraction of real eigenvalue mass that is positive.
    numerator = torch.clamp(real_parts, min=0).sum()
    denominator = real_parts.abs().sum()

    # Avoid division by zero when all real parts are zero.
    if denominator.item() == 0:
        return 0.0

    return float((numerator / denominator).item())


def full_ov_circuit(model, layer, head):
    # Extract the selected attention head's OV circuit in residual space.
    OV = ov_circuit(model.layers[layer][head])

    # Token embedding matrix: (V, d).
    E = model.wte.weight

    # Unembedding matrix: (V, d).
    U = model.unembed.weight

    # Compose embeddings, the OV circuit, and unembedding.
    # M[s, t] measures the logit contribution to token t from
    # attending to token s, ignoring layer normalization.
    M = (E @ OV.T @ U.T).detach()

    # Determine whether each row's largest entry lies on the diagonal.
    row_max = M.max(dim=1).values
    diagonal = M.diagonal()

    # Count diagonal entries that attain their row's maximum.
    diag_fraction = (diagonal == row_max).float().mean().item()

    return M, float(diag_fraction)

