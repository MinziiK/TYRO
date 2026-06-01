"""Parse TyroEnv PPO training log into a per-iteration summary table."""
from __future__ import annotations

import re
import sys
from pathlib import Path

METRICS = (
    "approach_tol", "start_pos_alpha", "contact_force_mean",
    "approach_A", "d_approach", "fsm_bonus",
    "ep_len_mean", "ep_rew_mean", "success_rate",
    "fps", "time_elapsed", "total_timesteps",
)


def parse_block(block: str) -> dict | None:
    out = {}
    for k in METRICS:
        m = re.search(rf"\|\s*{re.escape(k)}\s+\|\s*([-+\d.eE]+)\s*\|", block)
        if not m:
            return None
        out[k] = float(m.group(1))
    return out
EVAL_PATTERN = re.compile(
    r"Eval num_timesteps=(\d+), episode_reward=([\d.eE+-]+).*?"
    r"Episode length: ([\d.eE+-]+).*?Success rate: ([\d.eE+-]+)%",
    re.DOTALL,
)


def main() -> int:
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/phase1_fsm_easy_curriculum_v2.log")
    data = p.read_bytes()
    if data[:2] == b"\xff\xfe":
        text = data.decode("utf-16-le", errors="ignore")
    elif data[:2] == b"\xfe\xff":
        text = data.decode("utf-16-be", errors="ignore")
    else:
        text = data.decode("utf-8", errors="ignore")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"-{20,}", text)
    rows = []
    for b in blocks:
        d = parse_block(b)
        if d is not None:
            rows.append(d)
    print(f"# blocks_total={len(blocks)} matched={len(rows)}", file=sys.stderr)

    print(f"{'iter':>4} {'step':>8} {'min':>5} {'fps':>4} {'tol':>6} {'alpha':>6} "
          f"{'d_app':>6} {'ep_len':>6} {'ep_rew':>7} {'succ%':>5} {'appA':>5} "
          f"{'fsm':>7} {'cforce':>6}")
    for i, d in enumerate(rows, 1):
        print(
            f"{i:>4} {int(d['total_timesteps']):>8} {d['time_elapsed']/60:>5.0f} "
            f"{int(d['fps']):>4} {d['approach_tol']:>6.4f} {d['start_pos_alpha']:>6.3f} "
            f"{d['d_approach']:>6.3f} {int(d['ep_len_mean']):>6} {d['ep_rew_mean']:>7.1f} "
            f"{d['success_rate']*100:>5.1f} {d['approach_A']:>5.2f} "
            f"{d['fsm_bonus']:>7.4f} {int(d['contact_force_mean']):>6}"
        )

    print("\n--- EVAL (HOME + hard gate, deterministic) ---")
    print(f"{'step':>8} {'len':>6} {'rew':>7} {'succ%':>5}")
    for m in EVAL_PATTERN.finditer(text):
        s, r, l, su = m.groups()
        print(f"{int(s):>8} {int(float(l)):>6} {float(r):>7.1f} {float(su):>5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
