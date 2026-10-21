# Mechanistic Interpretability from Scratch: Find the Induction Circuit in a Tiny GPT

Induction heads are the first circuit anyone found inside a transformer and still the cleanest: a previous-token head in one layer writes 'the token before me was X' into the residual stream, and an induction head in the next layer reads that write to look up what followed X last time, then copies it. This project builds the whole investigation in pure PyTorch on a CPU. You design a task that can only be solved by induction (random tokens repeated at an unpredictable offset, so no positional shortcut works), write a two-layer attention-only GPT with every head's four matrices exposed, and watch in-context accuracy jump when the circuit forms. Then you find it three independent ways: from attention patterns (previous-token and induction scores on repeated-token prompts), from the weights alone (QK and OV circuits, the copying score, and the K-composition between layers), and from interventions (a logit-difference patching grid, zero versus mean versus resample ablation, and path patching that isolates the previous-token head's contribution through the induction head's keys). The final experiment scores the circuit for faithfulness, completeness and minimality, compares attribution patching with the exact grid, and shows the ablation that removes in-context learning while leaving the rest of the model intact, including the redundancy the model built for itself.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** make_batch
- [x] **2.** Head
- [x] **3.** AttnOnlyGPT
- [x] **4.** icl_accuracy
- [x] **5.** train_model
- [x] **6.** induction_score
- [x] **7.** head_scores
- [x] **8.** phase_change
- [x] **9.** qk_circuit
- [x] **10.** copying_score
- [x] **11.** k_composition
- [x] **12.** token_space_composition
- [x] **13.** make_pairs
- [x] **14.** patching_grid
- [x] **15.** ablation_effects
- [x] **16.** path_patch_report
- [x] **17.** direct_logit_attribution
- [x] **18.** circuit_scores
- [x] **19.** attribution_patching

---

Built on Deep-ML.
