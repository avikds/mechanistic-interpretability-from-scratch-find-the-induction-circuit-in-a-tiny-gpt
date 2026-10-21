"""
Mechanistic Interpretability from Scratch: Find the Induction Circuit in a Tiny GPT scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Mechanistic interpretability from scratch: find the induction circuit in a tiny GPT.

Story: a two-layer attention-only GPT is trained on random tokens repeated at an
unpredictable offset, a task only a lookup can solve. In-context accuracy jumps from
chance to near one in a few hundred steps, and at that moment one layer-0 head becomes
a previous-token head and the layer-1 heads become induction heads. The weights say
the same thing with no prompt at all: the induction heads copy, and the previous-token
head's write, read through the induction heads' keys, maps every token to itself.
Interventions close the case: patching restores the behaviour at the two positions the
mechanism predicts, path patching puts the effect in the keys, the circuit is faithful
and complete, the two induction heads back each other up, and removing the circuit
removes in-context learning.
"""
import torch


def main() -> None:
    torch.manual_seed(0)
    rep = induction_experiment(seed=0, steps=2000)
    roles, ev, ph = rep["roles"], rep["evidence"], rep["phase"]
    prev, inds = roles["previous"][0], roles["induction"]

    print("1. Training on a task only induction can solve")
    print(f"   in-context accuracy crossed 0.5 at step {ph['acc_step']}, the best induction score crossed 0.5 at step {ph['induction_step']}; final accuracy {ph['final_acc']:.3f}, final induction score {ph['final_induction']:.3f}")
    print("   The behaviour and the head appear together: there is no other way to solve the task.")

    print("\n2. Reading the patterns")
    for k, v in sorted(ev["scores"].items()):
        print(f"   head {k}: induction {v['induction']:.2f} | previous-token {v['prev_token']:.2f}")
    print(f"   previous-token head {prev}; induction heads {inds}")

    print("\n3. Reading the weights")
    for k in inds:
        print(f"   induction head {k}: OV copying score {ev['copying'][k]:.2f}, token-space OV diagonal fraction {ev['ov_diag'][k]:.2f}")
    print(f"   K-composition matrix (rows layer-0 heads, cols layer-1 heads): {ev['composition']} | random baseline {ev['baseline']:.3f}")
    for (earlier, later), (frac, margin) in sorted(ev["token_space"].items()):
        tag = "previous-token head" if earlier == prev else "other layer-0 head"
        print(f"   token-space composition {tag} {earlier} -> {later}: diagonal fraction {frac:.2f}, margin {margin:.1f}")
    print("   Through the induction heads' keys, the previous-token head's write says 'match the token before you': the mechanism, in the weights.")

    print("\n4. Intervening")
    print(f"   patching grid, fixed repeat length 8 (edit at position 6, previous-token head writes at 7, query at 14): top cells {rep['grid_top']}")
    print(f"   attribution patching recovers {rep['attrib_overlap']:.2f} of the top-3 cells")
    for head, eff in rep["ablations"].items():
        tag = "previous-token" if head == prev else "induction"
        print(f"   ablating {tag} head {head}: damage zero {eff['zero']:.2f} | mean {eff['mean']:.2f} | resample {eff['resample']:.2f}")
    p = rep["path"]
    print(f"   path patching {prev} -> {inds[0]}: restoration through queries {p['q']:.2f} | keys {p['k']:.2f} | values {p['v']:.2f} | all {p['qkv']:.2f}")
    print("   The previous-token head matters through the induction head's keys and nowhere else.")

    d = rep["dla"]
    print("\n5. Scoring the circuit")
    print("   direct logit attribution: " + ", ".join(f"{k}: {v:+.2f}" for k, v in d.items() if k not in ("total", "bias")) + f" (bias {d['bias']:+.2f}, total {d['total']:.2f})")
    s = rep["scores_full"]
    print(f"   circuit {rep['circuit']}: faithfulness {s['faithfulness']:.2f}, completeness {s['completeness']:.2f}, minimality {{{', '.join(f'{k}: {v:.2f}' for k, v in s['minimality'].items())}}}")
    if rep["scores_one"] is not None:
        o = rep["scores_one"]
        print(f"   with one induction head only: faithfulness {o['faithfulness']:.2f}, completeness {o['completeness']:.2f}")
        print("   The two induction heads split the copying between them, usually unequally: removing one costs part of the behaviour, removing both costs all of it, and the previous-token head is irreplaceable.")

    i = rep["icl"]
    print("\n6. What the circuit is for")
    print(f"   in-context accuracy: full model {i['full']:.3f} | induction heads zeroed {i['no_induction']:.3f} | previous-token head zeroed {i['no_prev']:.3f} | chance 0.031")
    print("   Remove either half of the circuit and in-context learning is gone; everything else the model does is untouched.")
    print("   Head indices change with the seed; the mechanism does not.")


if __name__ == "__main__":
    main()

