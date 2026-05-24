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
from models.jepa_wm import JepaEncoder, JepaPredictor, JepaActionEmbedder, SIGReg

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
    jepa_network_depth: int = 4
    jepa_skip_connections: int = 0
    use_relu: int = 0
    use_sig_reg: int = 1 # always used
    sig_reg_knots: int = 7
    sig_reg_weight: float = 0.09
    
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
    s_t_data = jnp.array(dataset["s_t"])
    a_t_data = jnp.array(dataset["a_t"])
    s_tp1_data = jnp.array(dataset["s_tp1"])
    
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
    
    # Shuffle once before splitting
    perm = jax.random.permutation(split_key, num_samples)
    s_t_data = s_t_data[perm]
    a_t_data = a_t_data[perm]
    s_tp1_data = s_tp1_data[perm]
    
    train_s_t = s_t_data[:num_train_samples]
    train_a_t = a_t_data[:num_train_samples]
    train_s_tp1 = s_tp1_data[:num_train_samples]
    
    eval_s_t = s_t_data[num_train_samples:]
    eval_a_t = a_t_data[num_train_samples:]
    eval_s_tp1 = s_tp1_data[num_train_samples:]
    
    print(f"Dataset loaded. train_samples: {num_train_samples}, eval_samples: {num_eval_samples}, obs_dim: {args.obs_dim}, action_size: {args.action_size}", flush=True)

    # JEPA Network Setup
    jepa_encoder = JepaEncoder(
        network_width=args.jepa_network_width,
        network_depth=args.jepa_network_depth,
        skip_connections=args.jepa_skip_connections,
        use_relu=args.use_relu,
    )
    jepa_action_embedder = JepaActionEmbedder(
        network_width=args.jepa_network_width,
        network_depth=args.jepa_network_depth,
        skip_connections=args.jepa_skip_connections,
        use_relu=args.use_relu,
    )
    jepa_predictor = JepaPredictor(
        network_width=args.jepa_network_width,
        network_depth=args.jepa_network_depth,
        skip_connections=args.jepa_skip_connections,
        use_relu=args.use_relu,
    )

    # SIGReg setup
    sig_reg = SIGReg(
        knots=args.sig_reg_knots,
        num_proj=64,
    )
    
    enc_key, embed_key, pred_key, sig_reg_key = jax.random.split(jepa_key, 4)
    encoder_params = jepa_encoder.init(enc_key, np.ones([1, args.obs_dim]))
    action_embedder_params = jepa_action_embedder.init(embed_key, np.ones([1, args.action_size]))
    # encoded representation has size 64 (hardcoded) # TODO make this an argument
    predictor_params = jepa_predictor.init(pred_key, np.ones([1, 64]), np.ones([1, 64]))
    sig_reg_params = sig_reg.init({"params": sig_reg_key, "proj": sig_reg_key}, np.ones([1, 64]))

    jepa_state = TrainState.create(
        apply_fn=None,
        params={
            "encoder": encoder_params,
            "predictor": predictor_params,
            "action_embedder": action_embedder_params,
        },
        tx=optax.adam(learning_rate=args.jepa_lr), # try adamw and add weight decay
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
    def eval_jepa(transitions, training_state, key):
        from models.jepa_wm import get_jepa_loss
        jepa_loss_fn = get_jepa_loss(
            args,
            jepa_encoder,
            jepa_predictor,
            jepa_action_embedder,
            sig_reg,
            sig_reg_params,
        )
        _, metrics = jepa_loss_fn(training_state.jepa_state.params, transitions, key)
        return metrics

    training_walltime = 0
    print("starting training....", flush=True)
    start_time = time.time()
    
    num_train_batches = num_train_samples // args.batch_size
    num_eval_batches = num_eval_samples // args.batch_size

    for ne in range(args.num_epochs):
        t = time.time()
        
        # Shuffle dataset
        key, epoch_key = jax.random.split(key)
        permutation = jax.random.permutation(epoch_key, num_train_samples)
        
        epoch_metrics = {"loss": [], "l2_loss": [], "sig_reg_loss": []}
        
        for i in range(num_train_batches):
            idx = permutation[i * args.batch_size : (i + 1) * args.batch_size]
            batch_s_t = train_s_t[idx]
            batch_a_t = train_a_t[idx]
            batch_s_tp1 = train_s_tp1[idx]
            
            batch_transitions = Transition(
                observation=batch_s_t,
                action=batch_a_t,
                reward=jnp.zeros(batch_s_t.shape[0]),
                discount=jnp.zeros(batch_s_t.shape[0]),
                extras={
                    "state": batch_s_t,
                    "next_state": batch_s_tp1,
                }
            )

            key, jepa_step_key = jax.random.split(key)
            training_state, metrics = update_jepa(batch_transitions, training_state, jepa_step_key)
            epoch_metrics["loss"].append(metrics["jepa_loss"])
            epoch_metrics["l2_loss"].append(metrics["jepa_l2_loss"])
            epoch_metrics["sig_reg_loss"].append(metrics["jepa_sig_reg_loss"])
            
        mean_loss = np.mean(epoch_metrics["loss"])
        mean_l2_loss = np.mean(epoch_metrics["l2_loss"])
        mean_sig_reg_loss = np.mean(epoch_metrics["sig_reg_loss"])
        
        # Evaluation
        eval_metrics = {"loss": [], "l2_loss": [], "sig_reg_loss": []}
        if num_eval_batches > 0:
            for i in range(num_eval_batches):
                idx = jnp.arange(i * args.batch_size, (i + 1) * args.batch_size)
                batch_s_t = eval_s_t[idx]
                batch_a_t = eval_a_t[idx]
                batch_s_tp1 = eval_s_tp1[idx]

                batch_transitions = Transition(
                    observation=batch_s_t,
                    action=batch_a_t,
                    reward=jnp.zeros(batch_s_t.shape[0]),
                    discount=jnp.zeros(batch_s_t.shape[0]),
                    extras={
                        "state": batch_s_t,
                        "next_state": batch_s_tp1,
                    }
                )

                key, jepa_step_key = jax.random.split(key)
                metrics = eval_jepa(batch_transitions, training_state, jepa_step_key)
                eval_metrics["loss"].append(metrics["jepa_loss"])
                eval_metrics["l2_loss"].append(metrics["jepa_l2_loss"])
                eval_metrics["sig_reg_loss"].append(metrics["jepa_sig_reg_loss"])

        mean_eval_loss = np.mean(eval_metrics["loss"]) if len(eval_metrics["loss"]) > 0 else 0.0
        mean_eval_l2_loss = np.mean(eval_metrics["l2_loss"]) if len(eval_metrics["l2_loss"]) > 0 else 0.0
        mean_eval_sig_reg_loss = np.mean(eval_metrics["sig_reg_loss"]) if len(eval_metrics["sig_reg_loss"]) > 0 else 0.0

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time

        log_metrics = {
            "training/walltime": training_walltime,
            "training/loss": mean_loss,
            "training/l2_loss": mean_l2_loss,
            "training/sig_reg_loss": mean_sig_reg_loss,
            "eval/loss": mean_eval_loss,
            "eval/l2_loss": mean_eval_l2_loss,
            "eval/sig_reg_loss": mean_eval_sig_reg_loss,
            "training/gradient_steps": training_state.gradient_steps.item()
        }

        print(f"epoch {ne} out of {args.num_epochs} complete. metrics: {log_metrics}", flush=True)

        if args.checkpoint:
            if ne < 5 or ne >= args.num_epochs - 5 or ne % 10 == 0:
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
