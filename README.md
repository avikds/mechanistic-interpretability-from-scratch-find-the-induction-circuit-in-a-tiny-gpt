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
- [x] **20.** build_model
- [x] **21.** circuit_evidence
- [x] **22.** induction_experiment

## Results

```
1. Training on a task only induction can solve
   in-context accuracy crossed 0.5 at step 600, the best induction score crossed 0.5 at step 600; final accuracy 0.890, final induction score 0.813
   The behaviour and the head appear together: there is no other way to solve the task.

2. Reading the patterns
   head (0, 0): induction 0.03 | previous-token 0.41
   head (0, 1): induction 0.06 | previous-token 0.04
   head (1, 0): induction 0.71 | previous-token 0.02
   head (1, 1): induction 0.81 | previous-token 0.02
   previous-token head (0, 0); induction heads [(1, 0), (1, 1)]

3. Reading the weights
   induction head (1, 0): OV copying score 0.63, token-space OV diagonal fraction 0.97
   induction head (1, 1): OV copying score 0.77, token-space OV diagonal fraction 0.94
   K-composition matrix (rows layer-0 heads, cols layer-1 heads): [[0.1865, 0.1873], [0.1507, 0.1585]] | random baseline 0.124
   token-space composition previous-token head (0, 0) -> (1, 0): diagonal fraction 1.00, margin 65.0
   token-space composition previous-token head (0, 0) -> (1, 1): diagonal fraction 1.00, margin 81.1
   token-space composition other layer-0 head (0, 1) -> (1, 0): diagonal fraction 0.50, margin 4.3
   token-space composition other layer-0 head (0, 1) -> (1, 1): diagonal fraction 0.38, margin 5.2
   Through the induction heads' keys, the previous-token head's write says 'match the token before you': the mechanism, in the weights.

4. Intervening
   patching grid, fixed repeat length 8 (edit at position 6, previous-token head writes at 7, query at 14): top cells [((0, 0), 7, 0.9959858655929565), ((1, 1), 14, 0.5390715003013611), ((1, 0), 14, 0.17324213683605194)]
   attribution patching recovers 0.67 of the top-3 cells
   ablating induction head prev: damage zero 1.01 | mean 1.00 | resample 1.02
   ablating induction head induction: damage zero 0.20 | mean 0.27 | resample 0.31
   path patching (0, 0) -> (1, 0): restoration through queries 0.00 | keys 0.20 | values -0.00 | all 0.20
   The previous-token head matters through the induction head's keys and nowhere else.

5. Scoring the circuit
   direct logit attribution: embed: -0.05, ('head', 0, 0): +0.03, ('head', 0, 1): +0.03, ('head', 1, 0): +1.30, ('head', 1, 1): +2.20 (bias +0.01, total 3.52)
   circuit [(0, 0), (1, 0), (1, 1)]: faithfulness 0.96, completeness 0.00, minimality {(0, 0): 0.96, (1, 0): 0.26, (1, 1): 0.69}
   with one induction head only: faithfulness 0.27, completeness 0.01
   The two induction heads split the copying between them, usually unequally: removing one costs part of the behaviour, removing both costs all of it, and the previous-token head is irreplaceable.

6. What the circuit is for
   in-context accuracy: full model 0.890 | induction heads zeroed 0.042 | previous-token head zeroed 0.051 | chance 0.031
   Remove either half of the circuit and in-context learning is gone; everything else the model does is untouched.
   Head indices change with the seed; the mechanism does not.
```
