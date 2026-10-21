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

# Step 11 - k_composition
def k_composition(QK_later, OV_earlier):
    # Measure how strongly the later head reads the earlier head's write.
    qk_norm = torch.linalg.norm(QK_later, ord="fro")
    ov_norm = torch.linalg.norm(OV_earlier, ord="fro")

    denominator = qk_norm * ov_norm

    # Handle zero-norm circuits safely.
    if denominator.item() == 0:
        return 0.0

    composed = QK_later @ OV_earlier
    score = torch.linalg.norm(composed, ord="fro") / denominator

    return float(score.item())


def composition_matrix(model):
    # Layer 0 contains the earlier (writing) heads.
    # Layer 1 contains the later (reading) heads.
    earlier_heads = model.layers[0]
    later_heads = model.layers[1]

    H0 = len(earlier_heads)
    H1 = len(later_heads)

    C = torch.empty(
        H0,
        H1,
        dtype=model.wte.weight.dtype,
        device=model.wte.weight.device,
    )

    for a, earlier in enumerate(earlier_heads):
        OV_earlier = ov_circuit(earlier)

        for b, later in enumerate(later_heads):
            QK_later = qk_circuit(later)
            C[a, b] = k_composition(QK_later, OV_earlier)

    return C


def random_composition_baseline(gen, d, d_head, trials):
    if trials <= 0:
        return 0.0

    scores = []

    for _ in range(trials):
        # Generate a reproducible seed for this trial.
        seed_i = int(
            torch.randint(0, 2**31, (1,), generator=gen).item()
        )

        # Keep the trial's random initialization isolated.
        with torch.random.fork_rng():
            torch.manual_seed(seed_i)

            # First head is the earlier writer; second is the later reader.
            first = Head(d, d_head)
            second = Head(d, d_head)

            # Match composition_matrix's reader–writer ordering.
            QK_later = qk_circuit(second)
            OV_earlier = ov_circuit(first)

            scores.append(
                k_composition(QK_later, OV_earlier)
            )

    return float(sum(scores) / len(scores))

# Step 12 - token_space_composition
def token_space_composition(model, earlier, later):
    # Retrieve the earlier writing head and the later reading head.
    earlier_layer, earlier_head = earlier
    later_layer, later_head = later

    earlier_module = model.layers[earlier_layer][earlier_head]
    later_module = model.layers[later_layer][later_head]

    # Compute the earlier head's OV circuit and the later head's QK circuit.
    OV_earlier = ov_circuit(earlier_module)
    QK_later = qk_circuit(later_module)

    # Token embedding matrix: (V, d).
    E = model.wte.weight

    # Compose the circuits in token space:
    # S[x, y] measures how strongly the later head scores token y
    # as a key when its query carries token x, through the earlier write.
    S = (E @ QK_later @ OV_earlier @ E.T).detach()

    # Fraction of rows whose maximum entry occurs on the diagonal.
    diagonal = S.diagonal()
    row_max = S.max(dim=1).values
    diag_fraction = (diagonal == row_max).float().mean().item()

    # Mean diagonal score minus mean off-diagonal score.
    V = S.shape[0]
    off_diag_mask = ~torch.eye(V, dtype=torch.bool, device=S.device)
    off_diagonal = S[off_diag_mask]

    if off_diagonal.numel() == 0:
        margin = 0.0
    else:
        margin = (diagonal.mean() - off_diagonal.mean()).item()

    return S, float(diag_fraction), float(margin)

# Step 13 - make_pairs
def make_pairs(B, gen, V=32, T=48, nmin=6, nmax=24):
    # Generate random sequences with a repeated prefix.
    tokens, n = make_batch(B, gen, V, T, nmin, nmax)

    # Start with a clean copy of every sequence.
    clean = tokens.clone()
    corrupt = tokens.clone()

    # Store the prediction position, correct answer, and incorrect alternative.
    pos = torch.empty(B, dtype=torch.long)
    answer = torch.empty(B, dtype=torch.long)
    wrong = torch.empty(B, dtype=torch.long)

    for i in range(B):
        k = int(n[i].item())

        # The final position in the second copy whose next token
        # remains inside that copy.
        pos[i] = 2 * k - 2

        # The correct next token is the token following the first
        # occurrence of the token at position k - 2.
        answer[i] = tokens[i, k - 1]

        # Select a guaranteed different token as the comparison target.
        wrong_offset = torch.randint(1, V, (1,), generator=gen).item()
        wrong[i] = (int(answer[i].item()) + wrong_offset) % V

        # Corrupt the original occurrence to disrupt the induction lookup.
        corrupt_offset = torch.randint(1, V, (1,), generator=gen).item()
        replacement = (
            int(tokens[i, k - 2].item()) + corrupt_offset
        ) % V

        corrupt[i, k - 2] = replacement

    return {
        "clean": clean,
        "corrupt": corrupt,
        "pos": pos,
        "answer": answer,
        "wrong": wrong,
        "n": n,
    }


@torch.no_grad()
def logit_diff(model, tokens, pos, answer, wrong, patch=None):
    # Run the model on the supplied prompts, optionally applying
    # the specified head-output patches.
    logits = model(tokens, patch=patch)

    # Construct batch indices on the same device as the logits.
    rows = torch.arange(logits.shape[0], device=logits.device)

    # Ensure the indexing tensors are on the correct device.
    pos = pos.to(device=logits.device, dtype=torch.long)
    answer = answer.to(device=logits.device, dtype=torch.long)
    wrong = wrong.to(device=logits.device, dtype=torch.long)

    # The logits at position t predict the token at position t + 1.
    correct_logits = logits[rows, pos, answer]
    incorrect_logits = logits[rows, pos, wrong]

    # Average correct-answer preference over all prompts.
    difference = correct_logits - incorrect_logits

    return float(difference.mean().item())

# Step 14 - patching_grid
@torch.no_grad()
def patching_grid(model, pairs):
    # Cache activations from the clean and corrupted prompts.
    clean_cache = {}
    corrupt_cache = {}

    model(pairs["clean"], cache=clean_cache)
    model(pairs["corrupt"], cache=corrupt_cache)

    # Measure the clean and corrupted baseline behaviors.
    m_clean = logit_diff(
        model,
        pairs["clean"],
        pairs["pos"],
        pairs["answer"],
        pairs["wrong"],
    )

    m_corrupt = logit_diff(
        model,
        pairs["corrupt"],
        pairs["pos"],
        pairs["answer"],
        pairs["wrong"],
    )

    # Normalization denominator for restoration scores.
    denominator = m_clean - m_corrupt

    # Determine the number of layers, heads, and prompt positions.
    L = len(model.layers)
    H = len(model.layers[0]) if L > 0 else 0
    T = pairs["clean"].shape[1]

    # Each row corresponds to one (layer, head) pair.
    grid = torch.zeros(
        (L * H, T),
        dtype=torch.float32,
        device=pairs["clean"].device,
    )
    labels = []

    row_idx = 0

    for l, layer in enumerate(model.layers):
        for h, _ in enumerate(layer):
            key = ("head", l, h)

            # Retrieve the corresponding clean and corrupted head outputs.
            clean_head = clean_cache[key]
            corrupt_head = corrupt_cache[key]

            labels.append((l, h))

            for p in range(T):
                # Start with the corrupted head output at every position.
                patched_head = corrupt_head.clone()

                # Restore this head's output at position p for every batch row.
                patched_head[:, p, :] = clean_head[:, p, :]

                # Patch only this head and this position.
                patch = {key: patched_head}

                # Evaluate the corrupted prompts with the intervention.
                m = logit_diff(
                    model,
                    pairs["corrupt"],
                    pairs["pos"],
                    pairs["answer"],
                    pairs["wrong"],
                    patch=patch,
                )

                # Measure normalized recovery of the clean behavior.
                if abs(denominator) < 1e-12:
                    restoration = 0.0
                else:
                    restoration = (m - m_corrupt) / denominator

                grid[row_idx, p] = restoration

            row_idx += 1

    return grid, labels

# Step 15 - ablation_effects
@torch.no_grad()
def ablation_effects(model, pairs, head, ref_pairs):
    # Identify the head to ablate.
    l, h = head
    head_key = ("head", l, h)

    B = pairs["clean"].shape[0]

    # The reference batch must contain at least B examples.
    if ref_pairs["clean"].shape[0] < B:
        raise ValueError(
            "ref_pairs must contain at least as many rows as pairs."
        )

    # Measure the clean and corrupted baseline logit differences.
    m_clean = logit_diff(
        model,
        pairs["clean"],
        pairs["pos"],
        pairs["answer"],
        pairs["wrong"],
    )

    m_corrupt = logit_diff(
        model,
        pairs["corrupt"],
        pairs["pos"],
        pairs["answer"],
        pairs["wrong"],
    )

    # Cache head outputs from the reference examples.
    ref_cache = {}
    model(ref_pairs["clean"], cache=ref_cache)

    # The cached head output has shape (B_ref, T, d).
    ref_output = ref_cache[head_key]

    # Use the first B reference activations for resampling.
    resample_output = ref_output[:B]

    # Zero ablation: replace the head's output at every position with zero.
    zero_output = torch.zeros_like(resample_output)

    # Mean ablation: replace every example with the reference batch's
    # mean head output, preserving the sequence position and feature.
    mean_output = ref_output.mean(dim=0, keepdim=True).expand(
        B, -1, -1
    )

    # Evaluate each intervention on the clean prompts.
    ablated_outputs = {
        "zero": zero_output,
        "mean": mean_output,
        "resample": resample_output,
    }

    # Normalize damage by the clean-to-corrupt behavioral difference.
    denominator = m_clean - m_corrupt
    effects = {}

    for mode, replacement in ablated_outputs.items():
        # Replace only the selected head's output.
        patch = {head_key: replacement}

        m_ablated = logit_diff(
            model,
            pairs["clean"],
            pairs["pos"],
            pairs["answer"],
            pairs["wrong"],
            patch=patch,
        )

        if abs(denominator) < 1e-12:
            damage = 0.0
        else:
            damage = (m_clean - m_ablated) / denominator

        effects[mode] = float(damage)

    return effects

