"""
extract_dataset_from_buffers.py

Extracts (s_t, a_t, s_tp1) transitions from one or more saved replay buffer
pickle files (produced by train.py's save_buffer_after_each_epoch feature) and
saves them to a single .npz file for use by train_jepa.py.

Buffer internal layout (43 floats per timestep per env):
  [0:31]  observation  (29 body-state dims + 2 goal dims appended by brax wrap)
  [31:39] action       (8 dims)
  [39]    reward
  [40]    discount
  [41]    truncation   (state_extras)
  [42]    seed         (state_extras – episode index used for trajectory stitching)

s_t / s_tp1 are the first 29 dims (ant body state, no goal), matching
args.obs_dim = 29 as set in train.py for all ant_maze environments.

Usage:
    # Single run, all epoch buffers:
    uv run extract_dataset_from_buffers.py \\
        --run_dirs wandb_crl/runs/ant_big_maze_1000_20260525-113051 \\
        --output transition_datasets/dataset.npz

    # Multiple runs merged:
    uv run extract_dataset_from_buffers.py \\
        --run_dirs wandb_crl/runs/run_A wandb_crl/runs/run_B \\
        --output transition_datasets/dataset_merged.npz

    # Keep only the later (better-explored) epoch buffers:
    uv run extract_dataset_from_buffers.py \\
        --run_dirs wandb_crl/runs/ant_big_maze_1000_20260525-113051 \\
        --min_epoch 10 \\
        --output transition_datasets/dataset_late.npz

    # Subsample to avoid a huge file:
    uv run extract_dataset_from_buffers.py \\
        --run_dirs wandb_crl/runs/ant_big_maze_1000_20260525-113051 \\
        --max_transitions 2000000 \\
        --output transition_datasets/dataset.npz
"""

import argparse
import glob
import os
import pickle
import re
import time

import numpy as np

# ── Layout constants (derived from empirical inspection of the .pkl files) ──
OBS_DIM_FULL  = 31   # full obs stored in buffer (29 body + 2 goal)
OBS_DIM_BODY  = 29   # first 29 dims = ant body state fed to JEPA encoder
ACTION_START  = 31
ACTION_END    = 39   # 8-dim action
ACTION_DIM    = 8
FLAT_SIZE     = 43   # total floats per timestep per env


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract JEPA training transitions from saved replay buffer pickles"
    )
    p.add_argument(
        "--run_dirs", nargs="+", required=True,
        help="One or more run directories containing replay_buffer_<epoch>.pkl files",
    )
    p.add_argument(
        "--output", default="transition_datasets/dataset.npz",
        help="Output .npz path",
    )
    p.add_argument(
        "--min_epoch", type=int, default=0,
        help="Only use buffers from epoch >= min_epoch (useful to skip early random-only data)",
    )
    p.add_argument(
        "--max_epoch", type=int, default=10_000,
        help="Only use buffers from epoch <= max_epoch",
    )
    p.add_argument(
        "--max_transitions", type=int, default=0,
        help="If > 0, randomly subsample down to this many transitions after merging",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for subsampling",
    )
    p.add_argument(
        "--deduplicate_epochs", action="store_true",
        help="If multiple epochs share the same buffer content (FIFO buffer is full "
             "and rolls), skip duplicate epochs to avoid data redundancy. "
             "Since each buffer is a FIFO of max_replay_size rows, consecutive "
             "full buffers overlap by (max_replay_size - new_rows) rows. "
             "Enable this to only take the NEWEST buffer from each run "
             "(which already contains all prior data when the buffer is full).",
    )
    return p.parse_args()


def epoch_from_path(path: str) -> int:
    """Extract epoch number from 'replay_buffer_<N>.pkl'."""
    m = re.search(r"replay_buffer_(\d+)\.pkl$", path)
    return int(m.group(1)) if m else -1


def extract_transitions_from_buffer(pkl_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one replay buffer pickle and return (s_t, a_t, s_tp1) arrays.

    The buffer stores data as a flat (max_replay_size, num_envs, FLAT_SIZE)
    array in FIFO order. Consecutive rows in the time dimension are consecutive
    environment steps, so row[t+1] is the next state for row[t] — UNLESS an
    episode boundary (truncation=1 or seed change) occurs.  We filter those out
    so that s_tp1 is always the genuine next observation.

    Returns:
        s_t   : (N, 29)  current body-state observations
        a_t   : (N,  8)  actions
        s_tp1 : (N, 29)  next body-state observations
    """
    with open(pkl_path, "rb") as f:
        buf = pickle.load(f)

    bs         = buf["buffer_state"]
    data       = np.array(bs.data)          # (T, E, 43)
    insert_pos = int(bs.insert_position)    # number of valid rows

    # Only use the rows that were actually written
    if insert_pos == 0:
        print(f"  [WARN] {pkl_path}: insert_position=0, skipping", flush=True)
        return np.zeros((0, OBS_DIM_BODY)), np.zeros((0, ACTION_DIM)), np.zeros((0, OBS_DIM_BODY))

    valid = data[:insert_pos]               # (T, E, 43)
    T, E, _ = valid.shape

    obs_t   = valid[:, :, :OBS_DIM_BODY]          # (T, E, 29)
    actions = valid[:, :, ACTION_START:ACTION_END] # (T, E, 8)
    # Field 41 = truncation (stores step-count-within-episode, NOT a 0/1 flag)
    # Field 42 = seed (0 or 1 epoch parity, flips each episode reset)
    seed_t  = valid[:, :, 42]                      # (T, E)

    # s_tp1 = obs at t+1.  We only have (T-1) consecutive pairs.
    obs_t1  = valid[1:, :, :OBS_DIM_BODY]         # (T-1, E, 29)

    # Build a mask: True where the transition (t → t+1) crosses NO episode boundary.
    # A boundary is detected when seed flips between consecutive timesteps.
    # (seed alternates 0→1→0→... each time the env auto-resets, which happens
    #  every episode_length steps inside brax's training wrapper.)
    seed_same  = (seed_t[:-1] == seed_t[1:])  # (T-1, E)  True = same episode
    valid_mask = seed_same                     # (T-1, E)

    # Flatten over (T-1, E) and apply mask
    s_t_all    = obs_t[:-1].reshape(-1, OBS_DIM_BODY)   # ((T-1)*E, 29)
    a_t_all    = actions[:-1].reshape(-1, ACTION_DIM)   # ((T-1)*E, 8)
    s_tp1_all  = obs_t1.reshape(-1, OBS_DIM_BODY)       # ((T-1)*E, 29)
    mask_flat  = valid_mask.reshape(-1)                 # ((T-1)*E,)

    s_t_out   = s_t_all[mask_flat]
    a_t_out   = a_t_all[mask_flat]
    s_tp1_out = s_tp1_all[mask_flat]

    total     = mask_flat.sum()
    pct       = 100.0 * total / mask_flat.size
    print(f"  {os.path.basename(pkl_path)}: {total:>8,} valid transitions "
          f"({pct:.1f}% of {mask_flat.size:,})", flush=True)
    return s_t_out, a_t_out, s_tp1_out


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    # ── Collect all matching .pkl paths ──────────────────────────────────
    all_paths = []
    for run_dir in args.run_dirs:
        pattern = os.path.join(run_dir, "replay_buffer_*.pkl")
        found   = sorted(glob.glob(pattern))
        if not found:
            print(f"[WARN] No replay_buffer_*.pkl found in {run_dir}", flush=True)
        all_paths.extend(found)

    if not all_paths:
        raise FileNotFoundError("No replay buffer files found. Check --run_dirs.")

    # ── Filter by epoch ───────────────────────────────────────────────────
    all_paths = [p for p in all_paths
                 if args.min_epoch <= epoch_from_path(p) <= args.max_epoch]

    if not all_paths:
        raise ValueError(f"No buffers in epoch range [{args.min_epoch}, {args.max_epoch}].")

    # ── Optionally keep only the latest epoch per run ─────────────────────
    if args.deduplicate_epochs:
        # Group by run directory, keep only max epoch per run
        from collections import defaultdict
        by_run: dict = defaultdict(list)
        for p in all_paths:
            by_run[os.path.dirname(p)].append(p)
        all_paths = [max(paths, key=epoch_from_path) for paths in by_run.values()]
        print(f"--deduplicate_epochs: keeping {len(all_paths)} buffer(s) "
              f"(one latest per run)", flush=True)

    # Sort for reproducibility
    all_paths = sorted(all_paths)
    print(f"\nProcessing {len(all_paths)} replay buffer file(s):", flush=True)

    # ── Extract transitions ───────────────────────────────────────────────
    t0          = time.time()
    s_t_parts   = []
    a_t_parts   = []
    s_tp1_parts = []

    for pkl_path in all_paths:
        s_t, a_t, s_tp1 = extract_transitions_from_buffer(pkl_path)
        s_t_parts.append(s_t)
        a_t_parts.append(a_t)
        s_tp1_parts.append(s_tp1)

    s_t   = np.concatenate(s_t_parts,   axis=0)
    a_t   = np.concatenate(a_t_parts,   axis=0)
    s_tp1 = np.concatenate(s_tp1_parts, axis=0)

    print(f"\nTotal before subsampling: {len(s_t):,} transitions", flush=True)

    # ── Optional deduplication by unique (s_t, a_t) ──────────────────────
    # (Consecutive full buffers overlap heavily; shuffling later is enough)

    # ── Subsample ─────────────────────────────────────────────────────────
    if args.max_transitions > 0 and len(s_t) > args.max_transitions:
        idx   = rng.choice(len(s_t), size=args.max_transitions, replace=False)
        s_t   = s_t[idx]
        a_t   = a_t[idx]
        s_tp1 = s_tp1[idx]
        print(f"Subsampled to {len(s_t):,} transitions", flush=True)

    # ── Shuffle ───────────────────────────────────────────────────────────
    perm   = rng.permutation(len(s_t))
    s_t    = s_t[perm]
    a_t    = a_t[perm]
    s_tp1  = s_tp1[perm]

    # ── Sanity check ──────────────────────────────────────────────────────
    xy = s_t[:, :2]
    print(f"\nXY coverage — x: [{xy[:,0].min():.2f}, {xy[:,0].max():.2f}]  "
          f"y: [{xy[:,1].min():.2f}, {xy[:,1].max():.2f}]", flush=True)
    print(f"Final shapes: s_t={s_t.shape}, a_t={a_t.shape}, s_tp1={s_tp1.shape}", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    np.savez_compressed(args.output, s_t=s_t, a_t=a_t, s_tp1=s_tp1)

    size_mb = os.path.getsize(args.output) / (1024 ** 2)
    elapsed = time.time() - t0
    print(f"\nSaved '{args.output}' ({size_mb:.1f} MB) in {elapsed:.1f}s", flush=True)
    print(f"\nTrain JEPA with:\n"
          f"  uv run train_jepa.py --dataset_path {args.output}", flush=True)


if __name__ == "__main__":
    main()
