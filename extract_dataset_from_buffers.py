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
    p.add_argument(
        "--sequential", action="store_true",
        help="Save data in trajectory-sequential (env-major) order required for "
             "multistep rollout training in train_jepa.py. Includes 'done' flags "
             "for clean episode-boundary detection. Incompatible with --max_transitions "
             "subsampling (which would break temporal ordering).",
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


def extract_transitions_sequential(
    pkl_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one replay buffer and return trajectory-sequential arrays.

    The buffer's (T, E, 43) layout is already per-env-sequential (row t+1 is
    the step after row t for each env). We expose this directly so that
    train_jepa.py's multistep windowing can build valid K-step windows.

    Episode boundaries are encoded in the returned `done` array (True where the
    seed field flips between consecutive timesteps).

    Returns:
        s_t   : (E*T, 29)  current body-state observations, env-major order
        a_t   : (E*T,  8)  actions
        s_tp1 : (E*T, 29)  next body-state observations
        done  : (E*T,)     bool — True at the LAST step of each episode

    Layout of the flat output:
        indices [0 .. T-1]       : env 0, steps 0..T-1
        indices [T .. 2T-1]      : env 1, steps 0..T-1
        ...
    """
    with open(pkl_path, "rb") as f:
        buf = pickle.load(f)

    bs         = buf["buffer_state"]
    data       = np.array(bs.data)       # (T_max, E, 43)
    insert_pos = int(bs.insert_position)

    if insert_pos == 0:
        print(f"  [WARN] {pkl_path}: insert_position=0, skipping", flush=True)
        empty = np.zeros((0,), dtype=bool)
        return (np.zeros((0, OBS_DIM_BODY)), np.zeros((0, ACTION_DIM)),
                np.zeros((0, OBS_DIM_BODY)), empty)

    valid = data[:insert_pos]             # (T, E, 43)
    T, E, _ = valid.shape

    obs     = valid[:, :, :OBS_DIM_BODY]           # (T, E, 29)
    actions = valid[:, :, ACTION_START:ACTION_END] # (T, E, 8)
    seed_t  = valid[:, :, 42]                      # (T, E)

    # done[t, e] = True means episode ended at step t for env e
    # (seed flips on the NEXT timestep, so done is detected as seed[t] != seed[t+1])
    # For the last timestep we conservatively mark done=True (no next row to compare).
    seed_flips = (seed_t[:-1] != seed_t[1:])       # (T-1, E)  True = boundary after t
    done_te = np.concatenate(
        [seed_flips, np.ones((1, E), dtype=bool)], axis=0
    )  # (T, E) — last row of each env is always marked done

    # Build s_tp1: for non-done steps use obs[t+1]; for done steps use obs[t]
    # (the true next obs after a reset isn't in this buffer row; keeping obs[t]
    #  avoids fabricating data — train_jepa.py will skip done transitions anyway)
    obs_tp1 = np.concatenate([obs[1:], obs[-1:]], axis=0)  # (T, E, 29)
    obs_tp1[done_te] = obs[done_te]  # mask out reset steps

    # Transpose (T, E, D) → (E, T, D) → (E*T, D)
    s_t_seq   = obs.transpose(1, 0, 2).reshape(-1, OBS_DIM_BODY)
    a_t_seq   = actions.transpose(1, 0, 2).reshape(-1, ACTION_DIM)
    s_tp1_seq = obs_tp1.transpose(1, 0, 2).reshape(-1, OBS_DIM_BODY)
    done_seq  = done_te.T.reshape(-1)  # (E*T,)

    valid_frac = 1.0 - done_seq.mean()
    print(
        f"  {os.path.basename(pkl_path)}: {T*E:>8,} steps "
        f"({E} envs × {T} steps, {100*valid_frac:.1f}% non-terminal)",
        flush=True,
    )
    return s_t_seq, a_t_seq, s_tp1_seq, done_seq


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
    t0 = time.time()

    if args.sequential:
        # ── Trajectory-sequential mode (for multistep rollout training) ───
        s_t_parts, a_t_parts, s_tp1_parts, done_parts = [], [], [], []

        for pkl_path in all_paths:
            s_t, a_t, s_tp1, done = extract_transitions_sequential(pkl_path)
            if len(s_t) == 0:
                continue
            s_t_parts.append(s_t)
            a_t_parts.append(a_t)
            s_tp1_parts.append(s_tp1)
            done_parts.append(done)

        # Concatenate buffers end-to-end.
        # Mark the join point of each buffer chunk as done so windows never
        # straddle two buffer files.
        done_arrays = []
        for d in done_parts:
            d = d.copy()
            d[-1] = True
            done_arrays.append(d)

        s_t   = np.concatenate(s_t_parts,   axis=0)
        a_t   = np.concatenate(a_t_parts,   axis=0)
        s_tp1 = np.concatenate(s_tp1_parts, axis=0)
        done  = np.concatenate(done_arrays,  axis=0)

        print(
            f"\nTotal (trajectory-sequential): {len(s_t):,} steps, "
            f"{100*(1-done.mean()):.1f}% non-terminal",
            flush=True,
        )

        # ── Episode-level subsampling ──────────────────────────────────────
        # Step-level subsampling would break temporal runs, so we sample
        # whole episodes instead.  Episodes are identified by done flags.
        if args.max_transitions > 0 and len(s_t) > args.max_transitions:
            print(
                f"Subsampling to ~{args.max_transitions:,} steps via "
                f"episode-level sampling (seed={args.seed})...",
                flush=True,
            )
            ep_end   = np.where(done)[0]                              # last step idx of each ep
            ep_start = np.concatenate([[0], ep_end[:-1] + 1])        # first step idx
            ep_len   = ep_end - ep_start + 1                         # length of each ep

            # Greedy random selection: shuffle episodes, add until budget full
            order = rng.permutation(len(ep_end))
            selected, total_sel = [], 0
            for i in order:
                if total_sel >= args.max_transitions:
                    break
                selected.append(i)
                total_sel += int(ep_len[i])

            # Sort so output stays temporally ordered within each episode block
            selected.sort()
            idx = np.concatenate([
                np.arange(ep_start[i], ep_end[i] + 1) for i in selected
            ])
            s_t   = s_t[idx]
            a_t   = a_t[idx]
            s_tp1 = s_tp1[idx]
            done  = done[idx]
            print(
                f"After episode sampling: {len(s_t):,} steps "
                f"({len(selected):,} episodes kept)",
                flush=True,
            )

        xy = s_t[:, :2]
        print(f"XY coverage — x: [{xy[:,0].min():.2f}, {xy[:,0].max():.2f}]  "
              f"y: [{xy[:,1].min():.2f}, {xy[:,1].max():.2f}]", flush=True)
        print(f"Final shapes: s_t={s_t.shape}, a_t={a_t.shape}, "
              f"s_tp1={s_tp1.shape}, done={done.shape}", flush=True)

        os.makedirs(
            os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
            exist_ok=True,
        )
        np.savez_compressed(
            args.output,
            s_t=s_t,
            a_t=a_t,
            s_tp1=s_tp1,
            done=done,
        )


    else:
        # ── Standard flat-pairs mode (original behaviour) ─────────────────
        s_t_parts, a_t_parts, s_tp1_parts = [], [], []

        for pkl_path in all_paths:
            s_t, a_t, s_tp1 = extract_transitions_from_buffer(pkl_path)
            s_t_parts.append(s_t)
            a_t_parts.append(a_t)
            s_tp1_parts.append(s_tp1)

        s_t   = np.concatenate(s_t_parts,   axis=0)
        a_t   = np.concatenate(a_t_parts,   axis=0)
        s_tp1 = np.concatenate(s_tp1_parts, axis=0)

        print(f"\nTotal before subsampling: {len(s_t):,} transitions", flush=True)

        if args.max_transitions > 0 and len(s_t) > args.max_transitions:
            idx   = rng.choice(len(s_t), size=args.max_transitions, replace=False)
            s_t   = s_t[idx]
            a_t   = a_t[idx]
            s_tp1 = s_tp1[idx]
            print(f"Subsampled to {len(s_t):,} transitions", flush=True)

        perm  = rng.permutation(len(s_t))
        s_t   = s_t[perm]
        a_t   = a_t[perm]
        s_tp1 = s_tp1[perm]

        xy = s_t[:, :2]
        print(f"\nXY coverage — x: [{xy[:,0].min():.2f}, {xy[:,0].max():.2f}]  "
              f"y: [{xy[:,1].min():.2f}, {xy[:,1].max():.2f}]", flush=True)
        print(f"Final shapes: s_t={s_t.shape}, a_t={a_t.shape}, s_tp1={s_tp1.shape}",
              flush=True)

        os.makedirs(
            os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
            exist_ok=True,
        )
        np.savez_compressed(args.output, s_t=s_t, a_t=a_t, s_tp1=s_tp1)

    size_mb = os.path.getsize(args.output) / (1024 ** 2)
    elapsed = time.time() - t0
    print(f"\nSaved '{args.output}' ({size_mb:.1f} MB) in {elapsed:.1f}s", flush=True)
    print(f"\nTrain JEPA with:\n"
          f"  uv run train_jepa.py --dataset_path {args.output}", flush=True)


if __name__ == "__main__":
    main()
