"""Smoke test: shaping reward path + bolt-index randomization across resets."""
from __future__ import annotations

import numpy as np

from src.config import make_env_config
from src.env import TyroEnv


def main() -> int:
    # Use stage 1 for the shaping-path inspection: contact-force termination is
    # disabled there, so we can observe several consecutive steps. The shaping
    # term is still computed in RewardBreakdown regardless of stage.
    cfg = make_env_config(stage=1, phase=1)
    print(f"stage       = 1  (contact_force termination disabled)")
    print(f"use_shaping = {cfg.use_shaping}  (False at stage 1 — shape_A/B still logged)")
    print(f"n_bolts     = {cfg.n_bolts}")
    env = TyroEnv(cfg=cfg, seed=0)

    # (1) Bolt randomization: 10 resets, count target indices.
    counts: dict[int, int] = {}
    for ep in range(10):
        _, info = env.reset(seed=100 + ep)
        idx = info["target_bolt_idx"]
        counts[idx] = counts.get(idx, 0) + 1
    print("\n[bolt randomization] target_bolt_idx over 10 resets:")
    for k in sorted(counts):
        print(f"  bolt {k}: {counts[k]}")
    print(f"  unique indices = {len(counts)}/10  (>1 means randomized)")

    # (2) Shaping path: first step shape ≈ 0, later steps potentially non-zero.
    # Use a tiny nudge action so we don't fly out of the workspace on step 1.
    env.reset(seed=42)
    header = ("step", "d_A", "d_B", "shape_A", "shape_B", "total", "term")
    print(
        "\n[shaping path] over 5 steps (tiny action):\n"
        f"  {header[0]:>4} {header[1]:>8} {header[2]:>8} "
        f"{header[3]:>10} {header[4]:>10} {header[5]:>10}  {header[6]}"
    )
    first_shape_was_zero = False
    later_shape_seen_nonzero = False
    for t in range(5):
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        a[0] = 0.05  # very small nudge
        _, _, term, trunc, info = env.step(a)
        rt = info["reward_terms"]
        term_str = info.get("termination", "-")
        print(
            f"  {t:>4d} {rt['d_A']:>8.4f} {rt['d_B']:>8.4f} "
            f"{rt['shape_A']:>+10.4f} {rt['shape_B']:>+10.4f} "
            f"{rt['total']:>+10.4f}  {term_str}"
        )
        if t == 0:
            first_shape_was_zero = (
                abs(rt["shape_A"]) < 1e-12 and abs(rt["shape_B"]) < 1e-12
            )
        elif abs(rt["shape_A"]) > 0 or abs(rt["shape_B"]) > 0:
            later_shape_seen_nonzero = True
        if term or trunc:
            print(f"  (episode ended at step {t}, termination={term_str})")
            break

    print(
        f"\n  first-step shape == 0 ?  {first_shape_was_zero}"
        f"\n  later step has non-zero shape ?  {later_shape_seen_nonzero}"
    )

    env.close()
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
