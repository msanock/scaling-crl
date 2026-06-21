import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb_osh
from etils import epath
from flax.training.train_state import TrainState
from wandb_osh.hooks import TriggerWandbSyncHook

import wandb
from models.jepa_wm import JepaEncoder, JepaPredictor, JepaActionEmbedder, SIGReg, JepaIDM

class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: dict


@dataclass
class Args:
    exp_name: str = "train_jepa_offline"
    seed: int = 1000
    dataset_path: str = "transition_datasets/dataset.npz"
    
    wandb_project_name: str = "jepa-crl"
    wandb_entity: str = "msanock-msanocki"
    wandb_mode: str = "online"
    wandb_dir: str = "wandb_jepa_only"
    wandb_group: str = "jepa_encoder"
    track: bool = True
    checkpoint: bool = True

    # JEPA Architecture specific arguments
    obs_dim: int = 0      # Inferred from dataset if 0
    action_size: int = 0  # Inferred from dataset if 0
    jepa_network_width: int = 256
    jepa_encoder_depth: int = 32
    jepa_action_embedder_depth: int = 4
    jepa_predictor_depth: int = 16
    jepa_skip_connections: int = 0
    use_relu: int = 0
    use_sig_reg: int = 1 # always used
    sig_reg_knots: int = 17
    sig_reg_num_proj: int = 1024
    sig_reg_weight: float = 0.09  # was 0.09 — lowered to prevent SIGReg from dominating
    # VICReg covariance/variance regularisation (Experiment 2)
    # var_loss_weight: penalises per-dim std < 1.0, preventing collapse
    # cov_loss_weight: penalises off-diagonal covariance → structural sparsity
    var_loss_weight: float = 0.0  # set to 1.0 to enable
    cov_loss_weight: float = 0.0  # set to 0.04 to enable
    # Temporal similarity and IDM losses (from paper eq. 12-13)
    sim_loss_weight: float = 0.0   # δ in paper; encourages smooth latent trajectories
    idm_loss_weight: float = 0.0   # ω in paper; predicts action from (z_t, z_{t+1})
    idm_network_width: int = 128   # hidden width of the IDM MLP
    # Multi-step rollout: K consecutive steps per training sample
    # rollout_length=1 → identical to the existing pair-based training
    rollout_length: int = 1        # K; requires sequential data (see dataset notes below)
    
    # Training
    num_epochs: int = 100
    batch_size: int = 256
    jepa_lr: float = 5e-5
    eval_split: float = 0.1
    # ema_tau: float = 0.01

@flax.struct.dataclass
class TrainingState:
    gradient_steps: jnp.ndarray
    jepa_state: TrainState
    # target_encoder_params: Any

def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)

def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))

if __name__ == "__main__":
    args = tyro.cli(Args)

    print(f"Loading dataset from {args.dataset_path}...", flush=True)
    dataset = np.load(args.dataset_path)
    s_t_data   = np.array(dataset["s_t"])
    a_t_data   = np.array(dataset["a_t"])
    s_tp1_data = np.array(dataset["s_tp1"])
    # Load done flags and metadata if present (new trajectory-sequential format)
    done_data      = np.array(dataset["done"],          dtype=bool) if "done"          in dataset else None
    dataset_n_envs = int(dataset["num_envs"])                       if "num_envs"      in dataset else None
    dataset_ep_len = int(dataset["episode_length"])                 if "episode_length" in dataset else None
    if dataset_n_envs is not None:
        print(f"Dataset metadata: num_envs={dataset_n_envs}, episode_length={dataset_ep_len}", flush=True)
    
    num_samples = s_t_data.shape[0]
    args.obs_dim = s_t_data.shape[1]
    args.action_size = a_t_data.shape[1]
    
    run_name = f"jepa_offline_{args.batch_size}_{args.seed}"
    print(f"run_name: {run_name}", flush=True)

    if args.track:
        if args.wandb_group == ".":
            args.wandb_group = None
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            group=args.wandb_group,
            dir=args.wandb_dir,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
        if args.wandb_mode == "offline":
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()

    if args.checkpoint:
        from datetime import datetime
        from pathlib import Path
        short_run_name = f"runs/jepa_offline_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        save_path = Path(args.wandb_dir) / Path(short_run_name)
        os.makedirs(save_path, exist_ok=True)

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, jepa_key, split_key = jax.random.split(key, 3)

    # Train / Eval Split
    num_eval_samples = int(num_samples * args.eval_split)
    num_train_samples = num_samples - num_eval_samples
    
    # -----------------------------------------------------------------
    # Multi-step windowing (must happen BEFORE the shuffle so that
    # consecutive indices are still temporally adjacent).
    # A window starting at i is valid only if s_tp1[i] ≈ s_t[i+1]
    # (i.e., no episode boundary between them).
    # When rollout_length=1 we keep the standard pair format.
    # -----------------------------------------------------------------
    K = args.rollout_length
    use_multistep = K > 1

    if use_multistep:
        print(f"Building windows of length K={K} from unshuffled dataset...", flush=True)
        # Determine episode-boundary continuity.
        # New format: use done flags directly (done[i]=True means episode ended at step i).
        # Old format: approximate via obs equality.
        if done_data is not None:
            # done[i]=True → stepping from i to i+1 crosses a reset; not continuous.
            continuity = ~done_data[:-1]  # (N-1,) True = safe to step through
        else:
            print("WARNING: 'done' not found in dataset; falling back to obs-equality check.", flush=True)
            continuity = np.all(
                np.abs(np.array(s_t_data[1:]) - np.array(s_tp1_data[:-1])) < 1e-4,
                axis=-1,
            )  # (N-1,)
        # A window of size K starting at i is valid iff
        # continuity[i], continuity[i+1], ..., continuity[i+K-2] are all True
        valid_starts = np.ones(num_samples - K, dtype=bool)
        for offset in range(K - 1):
            valid_starts &= continuity[offset : num_samples - K + offset]
        valid_indices = np.where(valid_starts)[0]  # window start positions
        print(f"Valid {K}-step windows: {len(valid_indices):,} / {num_samples - K:,}", flush=True)

        # Build windowed arrays (done in numpy for memory efficiency)
        win_states  = np.stack([np.array(s_t_data)[valid_indices + k] for k in range(K + 1)], axis=1)  # (W, K+1, D_obs)
        win_actions = np.stack([np.array(a_t_data)[valid_indices + k] for k in range(K)],     axis=1)  # (W, K, D_act)
        num_windows = len(valid_indices)
        print(f"win_states: {win_states.shape}, win_actions: {win_actions.shape}", flush=True)

        # Shuffle windows and split
        rng_win = np.random.default_rng(int(jax.random.randint(split_key, (), 0, 2**31 - 1)))
        perm_win = rng_win.permutation(num_windows)
        win_states  = win_states[perm_win]
        win_actions = win_actions[perm_win]

        num_eval_windows  = int(num_windows * args.eval_split)
        num_train_windows = num_windows - num_eval_windows

        train_win_states  = win_states[:num_train_windows]
        train_win_actions = win_actions[:num_train_windows]
        eval_win_states   = win_states[num_train_windows:]
        eval_win_actions  = win_actions[num_train_windows:]

        # Override sample counts used by the training loop
        num_train_samples = num_train_windows
        num_eval_samples  = num_eval_windows

    # Shuffle pairs (only used when rollout_length=1)
    if not use_multistep:
        perm = np.random.permutation(s_t_data.shape[0])
        s_t_data = s_t_data[perm]
        a_t_data = a_t_data[perm]
        s_tp1_data = s_tp1_data[perm]

    train_s_t   = s_t_data[:num_train_samples]
    train_a_t   = a_t_data[:num_train_samples]
    train_s_tp1 = s_tp1_data[:num_train_samples]

    eval_s_t   = s_t_data[num_train_samples:]
    eval_a_t   = a_t_data[num_train_samples:]
    eval_s_tp1 = s_tp1_data[num_train_samples:]
    
    print(f"Dataset loaded. train_samples: {num_train_samples}, eval_samples: {num_eval_samples}, obs_dim: {args.obs_dim}, action_size: {args.action_size}", flush=True)

    # JEPA Network Setup
    jepa_encoder = JepaEncoder(
        network_width=args.jepa_network_width,
        network_depth=args.jepa_encoder_depth,
        skip_connections=args.jepa_skip_connections,
        use_relu=args.use_relu,
    )
    jepa_action_embedder = JepaActionEmbedder(
        network_width=args.jepa_network_width,
        network_depth=args.jepa_action_embedder_depth,
        skip_connections=args.jepa_skip_connections,
        use_relu=args.use_relu,
    )
    jepa_predictor = JepaPredictor(
        network_width=args.jepa_network_width,
        network_depth=args.jepa_predictor_depth,
        skip_connections=args.jepa_skip_connections,
        use_relu=args.use_relu,
    )

    # SIGReg setup
    sig_reg = SIGReg(
        knots=args.sig_reg_knots,
        num_proj=args.sig_reg_num_proj,
    )
    
    enc_key, embed_key, pred_key, sig_reg_key, idm_key = jax.random.split(jepa_key, 5)
    encoder_params = jepa_encoder.init(enc_key, np.ones([1, args.obs_dim]))
    action_embedder_params = jepa_action_embedder.init(embed_key, np.ones([1, args.action_size]))
    # encoded representation has size 64 (hardcoded) # TODO make this an argument
    predictor_params = jepa_predictor.init(pred_key, np.ones([1, 64]), np.ones([1, 64]))
    sig_reg_params = sig_reg.init({"params": sig_reg_key, "proj": sig_reg_key}, np.ones([1, 64]))

    # IDM setup (always initialised; only contributes to loss when idm_loss_weight > 0)
    jepa_idm = JepaIDM(
        action_size=args.action_size,
        network_width=args.idm_network_width,
        use_relu=args.use_relu,
    )
    idm_params = jepa_idm.init(idm_key, np.ones([1, 64]), np.ones([1, 64]))

    jepa_state = TrainState.create(
        apply_fn=None,
        params={
            "encoder": encoder_params,
            "predictor": predictor_params,
            "action_embedder": action_embedder_params,
            "idm": idm_params,
        },
        tx=optax.adam(learning_rate=args.jepa_lr),
    )

    training_state = TrainingState(
        gradient_steps=jnp.zeros(()),
        jepa_state=jepa_state,
    )

    @jax.jit
    def update_jepa(transitions, training_state, key):
        from models.jepa_wm import get_jepa_loss
        jepa_loss_fn = get_jepa_loss(
            args,
            jepa_encoder,
            jepa_predictor,
            jepa_action_embedder,
            sig_reg,
            sig_reg_params,
            jepa_idm=jepa_idm,
            jepa_idm_params=training_state.jepa_state.params["idm"],
        )

        (loss, metrics), grad = jax.value_and_grad(jepa_loss_fn, has_aux=True)(
            training_state.jepa_state.params,
            transitions,
            key,
        )
        new_jepa_state = training_state.jepa_state.apply_gradients(grads=grad)

        training_state = training_state.replace(
            jepa_state=new_jepa_state,
            gradient_steps=training_state.gradient_steps + 1
        )
        return training_state, metrics

    @jax.jit
    def update_jepa_multistep(win_states, win_actions, training_state, key):
        """Multi-step update: win_states (B, K+1, D), win_actions (B, K, D_act)."""
        from models.jepa_wm import get_multistep_jepa_loss
        loss_fn = get_multistep_jepa_loss(
            args,
            jepa_encoder,
            jepa_predictor,
            jepa_action_embedder,
            sig_reg,
            sig_reg_params,
            jepa_idm=jepa_idm,
            jepa_idm_params=training_state.jepa_state.params["idm"],
        )

        (loss, metrics), grad = jax.value_and_grad(loss_fn, has_aux=True)(
            training_state.jepa_state.params,
            win_states,
            win_actions,
            key,
        )
        new_jepa_state = training_state.jepa_state.apply_gradients(grads=grad)
        training_state = training_state.replace(
            jepa_state=new_jepa_state,
            gradient_steps=training_state.gradient_steps + 1,
        )
        return training_state, metrics

    @jax.jit
    def eval_jepa(transitions, training_state, key):
        from models.jepa_wm import get_jepa_loss
        jepa_loss_fn = get_jepa_loss(
            args,
            jepa_encoder,
            jepa_predictor,
            jepa_action_embedder,
            sig_reg,
            sig_reg_params,
            jepa_idm=jepa_idm,
            jepa_idm_params=training_state.jepa_state.params["idm"],
        )
        _, metrics = jepa_loss_fn(training_state.jepa_state.params, transitions, key)
        return metrics

    @jax.jit
    def eval_jepa_multistep(win_states, win_actions, training_state, key):
        from models.jepa_wm import get_multistep_jepa_loss
        loss_fn = get_multistep_jepa_loss(
            args,
            jepa_encoder,
            jepa_predictor,
            jepa_action_embedder,
            sig_reg,
            sig_reg_params,
            jepa_idm=jepa_idm,
            jepa_idm_params=training_state.jepa_state.params["idm"],
        )
        _, metrics = loss_fn(training_state.jepa_state.params, win_states, win_actions, key)
        return metrics

    training_walltime = 0
    print("starting training....", flush=True)
    start_time = time.time()
    
    num_train_batches = num_train_samples // args.batch_size
    num_eval_batches = num_eval_samples // args.batch_size

    for ne in range(args.num_epochs):
        t = time.time()
        
        # Shuffle dataset using NumPy on CPU
        permutation = np.random.permutation(num_train_samples)
        epoch_metrics = {
            "loss": [], "l2_loss": [], "sig_reg_loss": [],
            "var_loss": [], "cov_loss": [], "sim_loss": [], "idm_loss": [],
        }

        for i in range(num_train_batches):
            key, jepa_step_key = jax.random.split(key)
            if use_multistep:
                idx = permutation[i * args.batch_size : (i + 1) * args.batch_size]
                b_win_states  = jnp.array(train_win_states[idx])
                b_win_actions = jnp.array(train_win_actions[idx])
                training_state, metrics = update_jepa_multistep(
                    b_win_states, b_win_actions, training_state, jepa_step_key
                )
            else:
                idx = permutation[i * args.batch_size : (i + 1) * args.batch_size]
                batch_s_t   = jnp.array(train_s_t[idx])
                batch_a_t   = jnp.array(train_a_t[idx])
                batch_s_tp1 = jnp.array(train_s_tp1[idx])
                batch_transitions = Transition(
                    observation=batch_s_t,
                    action=batch_a_t,
                    reward=jnp.zeros(batch_s_t.shape[0]),
                    discount=jnp.zeros(batch_s_t.shape[0]),
                    extras={"state": batch_s_t, "next_state": batch_s_tp1},
                )
                training_state, metrics = update_jepa(batch_transitions, training_state, jepa_step_key)

            epoch_metrics["loss"].append(metrics["jepa_loss"])
            epoch_metrics["l2_loss"].append(metrics["jepa_l2_loss"])
            epoch_metrics["sig_reg_loss"].append(metrics["jepa_sig_reg_loss"])
            epoch_metrics["var_loss"].append(metrics["jepa_var_loss"])
            epoch_metrics["cov_loss"].append(metrics["jepa_cov_loss"])
            epoch_metrics["sim_loss"].append(metrics["jepa_sim_loss"])
            epoch_metrics["idm_loss"].append(metrics["jepa_idm_loss"])

        mean_loss         = np.mean(epoch_metrics["loss"])
        mean_l2_loss      = np.mean(epoch_metrics["l2_loss"])
        mean_sig_reg_loss = np.mean(epoch_metrics["sig_reg_loss"])
        mean_var_loss     = np.mean(epoch_metrics["var_loss"])
        mean_cov_loss     = np.mean(epoch_metrics["cov_loss"])
        mean_sim_loss     = np.mean(epoch_metrics["sim_loss"])
        mean_idm_loss     = np.mean(epoch_metrics["idm_loss"])

        # Evaluation
        eval_metrics = {
            "loss": [], "l2_loss": [], "sig_reg_loss": [],
            "var_loss": [], "cov_loss": [], "sim_loss": [], "idm_loss": [],
        }
        if num_eval_batches > 0:
            for i in range(num_eval_batches):
                key, jepa_step_key = jax.random.split(key)
                if use_multistep:
                    idx = np.arange(i * args.batch_size, (i + 1) * args.batch_size)
                    b_win_states  = jnp.array(eval_win_states[idx])
                    b_win_actions = jnp.array(eval_win_actions[idx])
                    metrics = eval_jepa_multistep(b_win_states, b_win_actions, training_state, jepa_step_key)
                else:
                    idx = np.arange(i * args.batch_size, (i + 1) * args.batch_size)
                    batch_s_t   = jnp.array(eval_s_t[idx])
                    batch_a_t   = jnp.array(eval_a_t[idx])
                    batch_s_tp1 = jnp.array(eval_s_tp1[idx])
                    batch_transitions = Transition(
                        observation=batch_s_t,
                        action=batch_a_t,
                        reward=jnp.zeros(batch_s_t.shape[0]),
                        discount=jnp.zeros(batch_s_t.shape[0]),
                        extras={"state": batch_s_t, "next_state": batch_s_tp1},
                    )
                    metrics = eval_jepa(batch_transitions, training_state, jepa_step_key)

                eval_metrics["loss"].append(metrics["jepa_loss"])
                eval_metrics["l2_loss"].append(metrics["jepa_l2_loss"])
                eval_metrics["sig_reg_loss"].append(metrics["jepa_sig_reg_loss"])
                eval_metrics["var_loss"].append(metrics["jepa_var_loss"])
                eval_metrics["cov_loss"].append(metrics["jepa_cov_loss"])
                eval_metrics["sim_loss"].append(metrics["jepa_sim_loss"])
                eval_metrics["idm_loss"].append(metrics["jepa_idm_loss"])

        def _safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
        mean_eval_loss         = _safe_mean(eval_metrics["loss"])
        mean_eval_l2_loss      = _safe_mean(eval_metrics["l2_loss"])
        mean_eval_sig_reg_loss = _safe_mean(eval_metrics["sig_reg_loss"])
        mean_eval_var_loss     = _safe_mean(eval_metrics["var_loss"])
        mean_eval_cov_loss     = _safe_mean(eval_metrics["cov_loss"])
        mean_eval_sim_loss     = _safe_mean(eval_metrics["sim_loss"])
        mean_eval_idm_loss     = _safe_mean(eval_metrics["idm_loss"])

        # --- Inline PCA monitoring ---
        # Run the encoder on a single eval batch and compute explained variance.
        # Uses numpy SVD so it runs on CPU and is not JIT-compiled.
        pca_metrics = {}
        if num_eval_batches > 0:
            pca_src = eval_win_states[:args.batch_size, 0, :] if use_multistep else eval_s_t[:args.batch_size]
            pca_batch_s_t = np.array(pca_src)
            encoder_params_pca = training_state.jepa_state.params["encoder"]
            z_pca = np.array(jepa_encoder.apply(encoder_params_pca, pca_batch_s_t))  # (B, D)
            z_centered = z_pca - z_pca.mean(axis=0, keepdims=True)
            _, singular_values, _ = np.linalg.svd(z_centered, full_matrices=False)
            explained_var = (singular_values ** 2) / np.sum(singular_values ** 2)
            pca_metrics = {
                "pca/pc1_var": float(explained_var[0]),
                "pca/pc2_var": float(explained_var[1]),
                "pca/pc3_var": float(explained_var[2]),
                "pca/top3_var": float(explained_var[:3].sum()),
                "pca/top10_var": float(explained_var[:10].sum()),
                "pca/effective_rank": float(
                    np.exp(-np.sum(explained_var * np.log(explained_var + 1e-10)))
                ),
            }

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time

        log_metrics = {
            "training/walltime": training_walltime,
            "training/loss": mean_loss,
            "training/l2_loss": mean_l2_loss,
            "training/sig_reg_loss": mean_sig_reg_loss,
            "training/var_loss": mean_var_loss,
            "training/cov_loss": mean_cov_loss,
            "training/sim_loss": mean_sim_loss,
            "training/idm_loss": mean_idm_loss,
            "eval/loss": mean_eval_loss,
            "eval/l2_loss": mean_eval_l2_loss,
            "eval/sig_reg_loss": mean_eval_sig_reg_loss,
            "eval/var_loss": mean_eval_var_loss,
            "eval/cov_loss": mean_eval_cov_loss,
            "eval/sim_loss": mean_eval_sim_loss,
            "eval/idm_loss": mean_eval_idm_loss,
            "training/gradient_steps": training_state.gradient_steps.item(),
            **pca_metrics,
        }

        print(f"epoch {ne} out of {args.num_epochs} complete. metrics: {log_metrics}", flush=True)

        if args.checkpoint:
            if ne+1 % 10 == 0:
                params = (
                    training_state.jepa_state.params,
                )
                path = f"{save_path}/step_{int(training_state.gradient_steps)}.pkl"
                save_params(path, params)

        if args.track:
            wandb.log(log_metrics, step=ne)
            if args.wandb_mode == "offline":
                trigger_sync()

    if args.checkpoint:
        params = (
            training_state.jepa_state.params,
        )
        path = f"{save_path}/final.pkl"
        save_params(path, params)
        
        with open(f"{save_path}/args.pkl", "wb") as f:
            pickle.dump(args, f)
