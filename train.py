import os
import pickle
import random
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb_osh
from brax import envs
from brax.io import html
from etils import epath
from flax.training.train_state import TrainState
from wandb_osh.hooks import TriggerWandbSyncHook

import wandb
from buffer import TrajectoryUniformSamplingQueue
from evaluator import CrlEvaluator

from models.network_utils import lecun_unfirom, residual_block


@dataclass
class Args:
    exp_name: str = "train"
    seed: int = 1000
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "jepa-crl"
    wandb_entity: str = "msanock-msanocki"
    wandb_mode: str = "online"
    wandb_dir: str = "wandb_crl"
    wandb_group: str = "crl"
    capture_vis: bool = True
    capture_vis_every_n_epochs: int = 2
    vis_length: int = 1000
    checkpoint: bool = True

    # environment specific arguments
    env_id: str = "humanoid"  # "ant_big_maze" "humanoid_u_maze" "arm_binpick_hard"
    episode_length: int = 1000
    # to be filled in runtime
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0

    # Algorithm specific arguments
    total_env_steps: int = 100000000  # 50000000
    num_epochs: int = 100  # 50
    num_envs: int = 512
    eval_env_id: str = ""
    num_eval_envs: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    logsumexp_penalty_coeff: float = 0.1

    max_replay_size: int = 10000
    min_replay_size: int = 1000

    unroll_length: int = 62

    critic_network_width: int = 256
    actor_network_width: int = 256
    actor_depth: int = 32
    critic_depth: int = 32
    final_embedding_dim: int = 64
    actor_skip_connections: int = 0  # 0 for no skip connections, >= 0 means the frequency of skip connections (every N layers)
    critic_skip_connections: int = 0  # 0 for no skip connections, >= 0 means the frequency of skip connections (every N layers)

    num_episodes_per_env: int = 1  # recommended to keep at 1
    training_steps_multiplier: int = 1  # recommended to keep at 1
    use_all_batches: int = 0  # recommended to keep at 0
    num_sgd_batches_per_training_step: int = 800

    eval_actor: int = 0  # recommended to keep at 0
    # if 0, use deterministic actor for evaluation
    # if 1, use stochastic actor for evaluation
    # if 2, sample two actions and take the one with the higher Q value
    # if K >= 2, sample K actions and take the one with the highest Q value
    expl_actor: int = 1  # recommended to keep at 1
    # if 0, use deterministic actor for exploration/collecting data
    # if 1, use stochastic actor for exploration/collecting data
    # if 2, sample two actions and take the one with the higher Q value
    # if K >= 2, sample K actions and take the one with the highest Q value

    use_jepa: int = 0

    #jepa config
    jepa_checkpoint_path: str = ""
    jepa_continue_training: int = 1
    jepa_use_predictor_representation: int = 0
    jepa_concat_state_transition: int = 0
    jepa_gradient_scale: float = 0.0
    jepa_lr: float = 5e-5
    jepa_use_all_batches: int = 0
    offline_dataset_path: str = "transition_datasets/dataset_5M.npz"
    offline_ratio_start: float = 0.8
    offline_ratio_end: float = 0.2
    offline_decay_epochs: int = 40
    sig_reg_knots: int = 17
    sig_reg_num_proj: int = 1024
    sig_reg_weight: float = 0.09
    var_loss_weight: float = 0.0
    cov_loss_weight: float = 0.0
    sim_loss_weight: float = 0.0 
    jepa_encoder_depth: int = 32
    jepa_action_embedder_depth: int = 4
    jepa_predictor_depth: int = 16
    jepa_embedding_dim: int = 64
    jepa_actor: int = 0


    entropy_param: float = 0.5
    disable_entropy: int = 0
    use_relu: int = 0
    num_render: int = 10
    save_buffer: int = 0
    save_buffer_every_n_epochs: int = 0

    # to be filled in runtime
    env_steps_per_actor_step: int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps: int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps: int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch: int = 0
    """the number of training steps per epoch(computed in runtime)"""


class Actor(nn.Module):
    action_size: int
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        bias_init = nn.initializers.zeros

        # Initial layer
        x = nn.Dense(
            self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init
        )(x)
        x = normalize(x)
        x = activation(x)
        # Residual blocks
        for i in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        # Final layer
        mean = nn.Dense(
            self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init
        )(x)
        log_std = nn.Dense(
            self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init
        )(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner"""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
    jepa_state: Any = None


class Transition(NamedTuple):
    """Container for a transition"""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))



def create_jepa_critic(args, action_size, sa_key, g_key):
    from models.jepa_wm import JepaEncoder, JepaPredictor, JepaActionEmbedder, SA_encoderHead, SIGReg
    from models.classic_encoders import G_encoder

    jepa_encoder_key, jepa_predictor_key, sa_key = jax.random.split(sa_key, 3)

    jepa_state_encoder = JepaEncoder(
        network_width=args.critic_network_width,
        network_depth=args.jepa_encoder_depth,
        skip_connections=args.critic_skip_connections,
        use_relu=args.use_relu,
        output_dim=args.jepa_embedding_dim,
    )
    jepa_state_encoder_params = jepa_state_encoder.init(
        jepa_encoder_key, np.ones([1, args.obs_dim])
    )

    jepa_action_embedder = JepaActionEmbedder(
        network_width=args.critic_network_width,
        network_depth=args.jepa_action_embedder_depth,
        skip_connections=args.critic_skip_connections,
        use_relu=args.use_relu,
        output_dim=args.jepa_embedding_dim,
    )
    jepa_action_embedder_params = jepa_action_embedder.init(
        jepa_encoder_key, np.ones([1, action_size])
    )

    jepa_predictor = JepaPredictor(
        network_width=args.critic_network_width,
        network_depth=args.jepa_predictor_depth,
        skip_connections=args.critic_skip_connections,
        use_relu=args.use_relu,
        output_dim=args.jepa_embedding_dim,
    )
    jepa_predictor_params = jepa_predictor.init(
        jepa_predictor_key, np.ones([1, args.jepa_embedding_dim]), np.ones([1, args.jepa_embedding_dim])
    )

    sa_encoder = SA_encoderHead(
        network_width=args.critic_network_width,
        network_depth=args.critic_depth,
        skip_connections=args.critic_skip_connections,
        use_relu=args.use_relu,
        output_dim=args.final_embedding_dim,
    )

    if args.jepa_use_predictor_representation and not args.jepa_concat_state_transition:
        sa_encoder_input_dim = args.jepa_embedding_dim
    else:
        sa_encoder_input_dim = args.jepa_embedding_dim * 2
    
    sa_encoder_params = sa_encoder.init(
        sa_key, np.ones([1, sa_encoder_input_dim])
    )

    # Initialize SIGReg
    sig_reg = SIGReg(
        knots=args.sig_reg_knots,
        num_proj=args.sig_reg_num_proj,
    )
    sig_reg_key = jax.random.split(sa_key, 2)[1]
    sig_reg_params = sig_reg.init({"params": sig_reg_key, "proj": sig_reg_key}, np.ones([1, args.sig_reg_num_proj]))

    if args.jepa_continue_training:
        if args.jepa_checkpoint_path != "":
            loaded_params = load_params(args.jepa_checkpoint_path)[0]
            if "action_embedder" not in loaded_params:
                loaded_params = dict(loaded_params)
                loaded_params["action_embedder"] = jepa_action_embedder_params
            jepa_state = TrainState.create(
                apply_fn=None,
                params=flax.core.freeze(loaded_params),
                tx=optax.adam(learning_rate=args.jepa_lr),
            )
        else:
            jepa_state = TrainState.create(
                apply_fn=None,
                params=flax.core.freeze({"encoder": jepa_state_encoder_params, "predictor": jepa_predictor_params, "action_embedder": jepa_action_embedder_params }),
                tx=optax.adam(learning_rate=args.jepa_lr),
            )
    else:
        if not args.jepa_checkpoint_path:
            raise ValueError("JEPA checkpoint path is required when jepa_continue_training is False")
        
        loaded_params = load_params(args.jepa_checkpoint_path)[0]
        if "action_embedder" not in loaded_params:
            loaded_params = dict(loaded_params)
            loaded_params["action_embedder"] = jepa_action_embedder_params
        jepa_state = TrainState.create(
            apply_fn=None,
            params=flax.core.freeze(loaded_params), # index 0 is jepa_state params from train_jepa.py
            tx=optax.set_to_zero(),
        )

    g_encoder = G_encoder(
        network_width=args.critic_network_width,
        network_depth=args.critic_depth,
        skip_connections=args.critic_skip_connections,
        use_relu=args.use_relu,
        output_dim=args.final_embedding_dim,
    )
    g_encoder_params = g_encoder.init(
        g_key, np.ones([1, args.goal_end_idx - args.goal_start_idx])
    )

    critic_state = TrainState.create(
        apply_fn=None,
        params=flax.core.freeze({"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params}),
        tx=optax.adam(learning_rate=args.critic_lr),
    )

    return (
        critic_state,
        jepa_state,
        sa_encoder,
        g_encoder,
        sig_reg,
        sig_reg_params,
        jepa_state_encoder,
        jepa_predictor,
        jepa_action_embedder,
    )


def create_classic_critic(args, action_size, sa_key, g_key):
    from models.classic_encoders import G_encoder, SA_encoder

    sa_encoder = SA_encoder(
        network_width=args.critic_network_width,
        network_depth=args.critic_depth,
        skip_connections=args.critic_skip_connections,
        use_relu=args.use_relu,
        output_dim=args.final_embedding_dim,
    )
    sa_encoder_params = sa_encoder.init(
        sa_key, np.ones([1, args.obs_dim]), np.ones([1, action_size])
    )
    g_encoder = G_encoder(
        network_width=args.critic_network_width,
        network_depth=args.critic_depth,
        skip_connections=args.critic_skip_connections,
        use_relu=args.use_relu,
        output_dim=args.final_embedding_dim,
    )
    g_encoder_params = g_encoder.init(
        g_key, np.ones([1, args.goal_end_idx - args.goal_start_idx])
    )

    critic_state = TrainState.create(
        apply_fn=None,
        params=flax.core.freeze({"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params}),
        tx=optax.adam(learning_rate=args.critic_lr),
    )

    return critic_state, sa_encoder, g_encoder


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Print every arg
    print("Arguments:", flush=True)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}", flush=True)
    print("\n", flush=True)

    if args.use_jepa and args.offline_dataset_path:
        print(f"Loading offline dataset from {args.offline_dataset_path}...", flush=True)
        try:
            dataset = np.load(args.offline_dataset_path)
            offline_s_t = jnp.array(dataset["s_t"])
            offline_a_t = jnp.array(dataset["a_t"])
            offline_s_tp1 = jnp.array(dataset["s_tp1"])
            print(f"Offline dataset loaded. Samples: {offline_s_t.shape[0]}", flush=True)
        except Exception as e:
            print(f"Failed to load offline dataset: {e}", flush=True)
            raise e
    else:
        offline_s_t, offline_a_t, offline_s_tp1 = None, None, None

    args.env_steps_per_actor_step = args.num_envs * args.unroll_length
    print(f"env_steps_per_actor_step: {args.env_steps_per_actor_step}", flush=True)

    args.num_prefill_env_steps = args.min_replay_size * args.num_envs
    print(f"num_prefill_env_steps: {args.num_prefill_env_steps}", flush=True)

    args.num_prefill_actor_steps = np.ceil(args.min_replay_size / args.unroll_length)
    print(f"num_prefill_actor_steps: {args.num_prefill_actor_steps}", flush=True)

    args.num_training_steps_per_epoch = (
        args.total_env_steps - args.num_prefill_env_steps
    ) // (args.num_epochs * args.env_steps_per_actor_step)
    print(
        f"num_training_steps_per_epoch: {args.num_training_steps_per_epoch}", flush=True
    )

    run_name = f"{args.env_id}{'_' + args.eval_env_id if args.eval_env_id else ''}_{args.batch_size}_{args.total_env_steps}_nenvs:{args.num_envs}_criticwidth:{args.critic_network_width}_actorwidth:{args.actor_network_width}_criticdepth:{args.critic_depth}_actordepth:{args.actor_depth}_actorskip:{args.actor_skip_connections}_criticskip:{args.critic_skip_connections}_{args.seed}"
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
            monitor_gym=True,
            save_code=True,
        )

        if args.wandb_mode == "offline":
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()

    if args.checkpoint:
        from datetime import datetime
        from pathlib import Path

        short_run_name = (
            f"runs/{args.env_id}_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        save_path = Path(args.wandb_dir) / Path(short_run_name)
        os.makedirs(save_path, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key = jax.random.split(
        key, 7
    )

    def make_env(env_id=args.env_id):
        print(f"making env with env_id: {env_id}", flush=True)
        if env_id == "reacher":
            from envs.reacher import Reacher

            env = Reacher(
                backend="spring",
            )
            args.obs_dim = 10
            args.goal_start_idx = 4
            args.goal_end_idx = 7
        elif env_id == "pusher":
            from envs.pusher import Pusher

            env = Pusher(
                backend="spring",
            )
            args.obs_dim = 20
            args.goal_start_idx = 10
            args.goal_end_idx = 13
        elif env_id == "ant":
            from envs.ant import Ant

            env = Ant(
                backend="spring",
                exclude_current_positions_from_observation=False,
                terminate_when_unhealthy=True,
            )

            args.obs_dim = 29
            args.goal_start_idx = 0
            args.goal_end_idx = 2

        elif (
            "ant" in env_id and "maze" in env_id
        ):  # needed the add the ant check to differentiate with humanoid maze
            if "gen" not in env_id:
                from envs.ant_maze import AntMaze

                env = AntMaze(
                    backend="spring",
                    exclude_current_positions_from_observation=False,
                    terminate_when_unhealthy=True,
                    maze_layout_name=env_id[4:],
                )

                args.obs_dim = 29
                args.goal_start_idx = 0
                args.goal_end_idx = 2
            else:
                from envs.ant_maze_generalization import AntMazeGeneralization

                gen_idx = env_id.find("gen")
                maze_layout_name = env_id[4 : gen_idx - 1]
                generalization_config = env_id[gen_idx + 4 :]
                print(
                    f"maze_layout_name: {maze_layout_name}, generalization_config: {generalization_config}",
                    flush=True,
                )
                env = AntMazeGeneralization(
                    backend="spring",
                    exclude_current_positions_from_observation=False,
                    terminate_when_unhealthy=True,
                    maze_layout_name=maze_layout_name,
                    generalization_config=generalization_config,
                )

                args.obs_dim = 29
                args.goal_start_idx = 0
                args.goal_end_idx = 2

        elif env_id == "ant_ball":
            from envs.ant_ball import AntBall

            env = AntBall(
                backend="spring",
                exclude_current_positions_from_observation=False,
                terminate_when_unhealthy=True,
            )

            args.obs_dim = 31
            args.goal_start_idx = 28
            args.goal_end_idx = 30

        elif env_id == "ant_push":
            from envs.ant_push import AntPush

            env = AntPush(
                backend="mjx",
            )

            args.obs_dim = 31
            args.goal_start_idx = 0
            args.goal_end_idx = 2

        elif env_id == "humanoid":
            from envs.humanoid import Humanoid

            env = Humanoid(
                backend="spring",
                exclude_current_positions_from_observation=False,
                terminate_when_unhealthy=True,
            )

            args.obs_dim = 268
            args.goal_start_idx = 0
            args.goal_end_idx = 3

        elif "humanoid" in env_id and "maze" in env_id:
            from envs.humanoid_maze import HumanoidMaze

            env = HumanoidMaze(backend="spring", maze_layout_name=env_id[9:])

            args.obs_dim = 268
            args.goal_start_idx = 0
            args.goal_end_idx = 3

        elif env_id == "arm_reach":
            from envs.manipulation.arm_reach import ArmReach

            env = ArmReach(
                backend="mjx",
            )

            args.obs_dim = 13
            args.goal_start_idx = 7
            args.goal_end_idx = 10

        elif env_id == "arm_binpick_easy":
            from envs.manipulation.arm_binpick_easy import ArmBinpickEasy

            env = ArmBinpickEasy(
                backend="mjx",
            )

            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3

        elif env_id == "arm_binpick_hard":
            from envs.manipulation.arm_binpick_hard import ArmBinpickHard

            env = ArmBinpickHard(
                backend="mjx",
            )

            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3

        elif env_id == "arm_binpick_easy_EEF":
            from envs.manipulation.arm_binpick_easy_EEF import ArmBinpickEasyEEF

            env = ArmBinpickEasyEEF(
                backend="mjx",
            )

            args.obs_dim = 11
            args.goal_start_idx = 0
            args.goal_end_idx = 3

        elif "arm_grasp" in env_id:  # either arm_grasp or arm_grasp_0.5, etc
            from envs.manipulation.arm_grasp import ArmGrasp

            cube_noise_scale = float(env_id[10:]) if len(env_id) > 9 else 0.3
            env = ArmGrasp(
                cube_noise_scale=cube_noise_scale,
                backend="mjx",
            )

            args.obs_dim = 23
            args.goal_start_idx = 16
            args.goal_end_idx = 23

        elif env_id == "arm_push_easy":
            from envs.manipulation.arm_push_easy import ArmPushEasy

            env = ArmPushEasy(
                backend="mjx",
            )

            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3

        elif env_id == "arm_push_hard":
            from envs.manipulation.arm_push_hard import ArmPushHard

            env = ArmPushHard(
                backend="mjx",
            )

            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3

        else:
            raise NotImplementedError

        return env

    env = make_env()
    env = envs.training.wrap(
        env,
        episode_length=args.episode_length,
    )

    obs_size = env.observation_size
    action_size = env.action_size
    env_keys = jax.random.split(env_key, args.num_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step = jax.jit(env.step)

    print(f"obs_size: {obs_size}, action_size: {action_size}", flush=True)

    if not args.eval_env_id:
        args.eval_env_id = args.env_id

    # make eval env
    eval_env = make_env(args.eval_env_id)
    eval_env = envs.training.wrap(
        eval_env,
        episode_length=args.episode_length,
    )
    eval_env_keys = jax.random.split(eval_env_key, args.num_envs)
    eval_env_state = jax.jit(eval_env.reset)(eval_env_keys)
    eval_env.step = jax.jit(eval_env.step)

    # Network setup
    # Actor
    actor = Actor(
        action_size=action_size,
        network_width=args.actor_network_width,
        network_depth=args.actor_depth,
        skip_connections=args.actor_skip_connections,
        use_relu=args.use_relu,
    )
    actor_input_dim = args.jepa_embedding_dim + args.final_embedding_dim if args.jepa_actor else obs_size
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, actor_input_dim])),
        tx=optax.adam(learning_rate=args.actor_lr),
    )

    jepa_state = None
    sig_reg = None
    sig_reg_params = None
    jepa_state_encoder = None
    jepa_predictor = None
    jepa_action_embedder = None

    # Critic
    if args.use_jepa:
        (
            critic_state,
            jepa_state,
            sa_encoder,
            g_encoder,
            sig_reg,
            sig_reg_params,
            jepa_state_encoder,
            jepa_predictor,
            jepa_action_embedder,
        ) = create_jepa_critic(args, action_size, sa_key, g_key)
    else:
        critic_state, sa_encoder, g_encoder = create_classic_critic(
            args, action_size, sa_key, g_key
        )

    # Entropy coefficient
    target_entropy = (
        -args.entropy_param * action_size
    )  # action_size = 8 for ant, 17 for humanoid, etc
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": log_alpha},
        tx=optax.adam(learning_rate=args.alpha_lr),
    )

    # Trainstate
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
        jepa_state=jepa_state,
    )

    # Replay Buffer
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))

    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        extras={
            "state_extras": {
                "truncation": 0.0,
                "seed": 0.0,
            }
        },
    )

    def jit_wrap(buffer):
        buffer.insert_internal = jax.jit(buffer.insert_internal)
        buffer.sample_internal = jax.jit(buffer.sample_internal)
        return buffer

    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=args.max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=args.batch_size,
            num_envs=args.num_envs,
            episode_length=args.episode_length,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        if args.jepa_actor:
            encoded_state = jepa_state_encoder.apply(
                training_state.jepa_state.params["encoder"],
                env_state.obs[:, : args.obs_dim]
            )
            goal_embed = g_encoder.apply(
                training_state.critic_state.params["g_encoder"],
                env_state.obs[:, args.obs_dim :]
            )
            actor_input = jnp.concatenate((encoded_state, goal_embed), axis=-1)
        else:
            actor_input = env_state.obs
        means, _ = actor.apply(training_state.actor_state.params, actor_input)
        actions = nn.tanh(means)

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}

        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    def actor_step(training_state, env, env_state, key, extra_fields):
        if args.jepa_actor:
            encoded_state = jepa_state_encoder.apply(
                training_state.jepa_state.params["encoder"],
                env_state.obs[:, : args.obs_dim]
            )
            goal_embed = g_encoder.apply(
                training_state.critic_state.params["g_encoder"],
                env_state.obs[:, args.obs_dim :]
            )
            actor_input = jnp.concatenate((encoded_state, goal_embed), axis=-1)
        else:
            actor_input = env_state.obs
            
        means, log_stds = actor.apply(training_state.actor_state.params, actor_input)
        stds = jnp.exp(log_stds)
        actions = nn.tanh(
            means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        )

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}

        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    def multi_sample_actor_step(training_state, env, env_state, key, K, extra_fields):
        # Get K sets of actions from the actor
        keys = jax.random.split(key, K)
        
        if args.jepa_actor:
            encoded_state = jepa_state_encoder.apply(
                training_state.jepa_state.params["encoder"],
                env_state.obs[:, : args.obs_dim]
            )
            goal_embed = g_encoder.apply(
                training_state.critic_state.params["g_encoder"],
                env_state.obs[:, args.obs_dim :]
            )
            actor_input = jnp.concatenate((encoded_state, goal_embed), axis=-1)
        else:
            actor_input = env_state.obs
            
        means, log_stds = actor.apply(training_state.actor_state.params, actor_input)
        stds = jnp.exp(log_stds)

        actions = jnp.stack(
            [
                nn.tanh(
                    means
                    + stds * jax.random.normal(k, shape=means.shape, dtype=means.dtype)
                )
                for k in keys
            ]
        )

        state = env_state.obs[:, : args.obs_dim]
        goal = env_state.obs[:, args.obs_dim :]

        sa_reprs = jax.vmap(
            lambda a: sa_encoder.apply(
                training_state.critic_state.params["sa_encoder"], state, a
            )
        )(actions)

        g_repr = g_encoder.apply(training_state.critic_state.params["g_encoder"], goal)

        q_values = -jnp.sqrt(jnp.sum((sa_reprs - g_repr) ** 2, axis=-1))

        best_action_idx = jnp.argmax(q_values, axis=0)
        best_actions = jnp.take_along_axis(
            actions, best_action_idx[None, :, None], axis=0
        )[0]

        # Step environment with best actions
        nstate = env.step(env_state, best_actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}

        return nstate, Transition(
            observation=env_state.obs,
            action=best_actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    @jax.jit
    def get_experience(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):  # conducts a single actor step in environment
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            if args.expl_actor == 1:
                env_state, transition = actor_step(
                    training_state,
                    env,
                    env_state,
                    current_key,
                    extra_fields=("truncation", "seed"),
                )
            elif args.expl_actor == 0:
                env_state, transition = deterministic_actor_step(
                    training_state, env, env_state, extra_fields=("truncation", "seed")
                )
            else:
                env_state, transition = multi_sample_actor_step(
                    training_state,
                    env,
                    env_state,
                    current_key,
                    args.expl_actor,
                    extra_fields=("truncation", "seed"),
                )
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(
            f, (env_state, key), (), length=args.unroll_length
        )

        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(
                training_state,
                env_state,
                buffer_state,
                key,
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + args.env_steps_per_actor_step,
            )
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=args.num_prefill_actor_steps,
        )[0]

    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        actor_batch_size = args.batch_size
        transitions = jax.tree_util.tree_map(
            lambda x: x[:actor_batch_size], transitions
        )

        def actor_loss(actor_params, critic_params, jepa_params, log_alpha, transitions, key):
            obs = (
                transitions.observation
            )  # expected_shape = batch_size, obs_size + goal_size
            state = obs[:, : args.obs_dim]
            future_state = transitions.extras["future_state"]
            goal = future_state[:, args.goal_start_idx : args.goal_end_idx]
            
            if args.jepa_actor:
                encoded_state = jax.lax.stop_gradient(jepa_state_encoder.apply(
                    jepa_params["encoder"],
                    state
                ))
                # TODO: Using g_encoder from critic, might be a bad idea
                goal_embed = jax.lax.stop_gradient(g_encoder.apply(
                    critic_params["g_encoder"],
                    goal
                ))
                actor_input = jnp.concatenate([encoded_state, goal_embed], axis=1)
            else:
                actor_input = jnp.concatenate([state, goal], axis=1)

            means, log_stds = actor.apply(actor_params, actor_input)
            stds = jnp.exp(log_stds)
            x_ts = means + stds * jax.random.normal(
                key, shape=means.shape, dtype=means.dtype
            )
            action = nn.tanh(x_ts)
            log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)  # dimension = B

            # TODO: get_sa_repr and get_g_repr to other functions
            sa_encoder_params, g_encoder_params = (
                critic_params["sa_encoder"],
                critic_params["g_encoder"],
            )
            
            if args.use_jepa:
                z_s = jepa_state_encoder.apply(jepa_params["encoder"], state)
                if args.jepa_use_predictor_representation:
                    a_embed = jepa_action_embedder.apply(jepa_params["action_embedder"], action)
                    z_s_tp1 = jepa_predictor.apply(jepa_params["predictor"], z_s, a_embed)
                    x = jnp.concatenate([z_s, z_s_tp1], axis=-1) if args.jepa_concat_state_transition else z_s_tp1
                else:
                    a_embed = jepa_action_embedder.apply(jepa_params["action_embedder"], action)
                    x = jnp.concatenate([z_s, a_embed], axis=-1)
                sa_repr = sa_encoder.apply(sa_encoder_params, x)
            else:
                sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
                
            g_repr = g_encoder.apply(g_encoder_params, goal)

            qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

            if args.disable_entropy:
                actor_loss = -jnp.mean(qf_pi)
            else:
                actor_loss = jnp.mean(jnp.exp(log_alpha) * log_prob - (qf_pi))

            return actor_loss, log_prob

        def alpha_loss(alpha_params, log_prob):
            alpha = jnp.exp(alpha_params["log_alpha"])
            alpha_loss = alpha * jnp.mean(
                jax.lax.stop_gradient(-log_prob - target_entropy)
            )
            return jnp.mean(alpha_loss)

        (actorloss, log_prob), actor_grad = jax.value_and_grad(
            actor_loss, has_aux=True
        )(
            training_state.actor_state.params,
            training_state.critic_state.params,
            training_state.jepa_state.params if args.use_jepa else None,
            training_state.alpha_state.params["log_alpha"],
            transitions,
            key,
        )
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        alphaloss, alpha_grad = jax.value_and_grad(alpha_loss)(
            training_state.alpha_state.params, log_prob
        )
        new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

        training_state = training_state.replace(
            actor_state=new_actor_state, alpha_state=new_alpha_state
        )

        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actorloss,
            "alph_aloss": alphaloss,
            "log_alpha": training_state.alpha_state.params["log_alpha"],
        }

        return training_state, metrics

    
    @jax.jit
    def update_critic(transitions, training_state, key):
        critic_batch_size = args.batch_size
        transitions = jax.tree_util.tree_map(
            lambda x: x[:critic_batch_size], transitions
        )

        if args.use_jepa:
            def critic_loss(critic_params, jepa_params, transitions, key):
                s = transitions.observation[:, : args.obs_dim]
                a = transitions.action
                
                z_s = jepa_state_encoder.apply(jepa_params["encoder"], s)
                if args.jepa_use_predictor_representation:
                    a_embed = jepa_action_embedder.apply(jepa_params["action_embedder"], a)
                    z_s_tp1 = jepa_predictor.apply(jepa_params["predictor"], z_s, a_embed)
                    x = jnp.concatenate([z_s, z_s_tp1], axis=-1) if args.jepa_concat_state_transition else z_s_tp1
                else:
                    a_embed = jepa_action_embedder.apply(jepa_params["action_embedder"], a)
                    x = jnp.concatenate([z_s, a_embed], axis=-1)
                    
                sa_repr = sa_encoder.apply(critic_params["sa_encoder"], x)
                g_repr = g_encoder.apply(critic_params["g_encoder"], transitions.observation[:, args.obs_dim :])
                
                # InfoNCE
                logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
                loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
                
                logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
                loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)
                
                I, correct, logits_pos, logits_neg = jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), jnp.zeros(1)
                return loss, (logsumexp, I, correct, logits_pos, logits_neg)

            (loss, (logsumexp, I, correct, logits_pos, logits_neg)), (critic_grad, jepa_grad) = jax.value_and_grad(critic_loss, argnums=(0, 1), has_aux=True)(
                training_state.critic_state.params, training_state.jepa_state.params, transitions, key
            )
            
            if args.jepa_gradient_scale > 0.0:
                jepa_grad = jax.tree_util.tree_map(lambda g: g * args.jepa_gradient_scale, jepa_grad)
                new_jepa_state = training_state.jepa_state.apply_gradients(grads=jepa_grad)
                training_state = training_state.replace(jepa_state=new_jepa_state)

            new_critic_state = training_state.critic_state.apply_gradients(grads=critic_grad)
            training_state = training_state.replace(critic_state=new_critic_state)

        else:
            from models.classic_encoders import get_classic_critic_loss
            critic_loss = get_classic_critic_loss(args, sa_encoder, g_encoder)
    
            (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = (
                jax.value_and_grad(critic_loss, has_aux=True)(
                    training_state.critic_state.params, transitions, key
                )
            )
            new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
            training_state = training_state.replace(critic_state=new_critic_state)

        metrics = {
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logsumexp": logsumexp.mean(),
            "critic_loss": loss,
        }

        return training_state, metrics

    if args.use_jepa and args.jepa_continue_training:
        @jax.jit
        def update_jepa(transitions, training_state, key, offline_batch, offline_ratio):
            from models.jepa_wm import get_jepa_loss

            s_t = transitions.extras["state"]
            a_t = transitions.action
            s_tp1 = transitions.extras["next_state"]

            key, mix_key = jax.random.split(key)
            if args.offline_dataset_path:
                offline_s_t_batch, offline_a_t_batch, offline_s_tp1_batch = offline_batch
                batch_size = s_t.shape[0]
                mask = jax.random.bernoulli(mix_key, p=offline_ratio, shape=(batch_size,))
                mask_expanded_s = jnp.expand_dims(mask, axis=1)
                mask_expanded_a = jnp.expand_dims(mask, axis=1)

                s_t = jnp.where(mask_expanded_s, offline_s_t_batch, s_t)
                a_t = jnp.where(mask_expanded_a, offline_a_t_batch, a_t)
                s_tp1 = jnp.where(mask_expanded_s, offline_s_tp1_batch, s_tp1)

            mixed_extras = transitions.extras.copy()
            mixed_extras["state"] = s_t
            mixed_extras["next_state"] = s_tp1
            mixed_transitions = transitions._replace(action=a_t, extras=mixed_extras)

            jepa_loss_fn = get_jepa_loss(
                args,
                jepa_state_encoder,
                jepa_predictor,
                jepa_action_embedder,
                sig_reg,
                sig_reg_params,
            )

            (loss, metrics), grad = jax.value_and_grad(jepa_loss_fn, has_aux=True)(
                training_state.jepa_state.params,
                mixed_transitions,
                key,
            )
            new_jepa_state = training_state.jepa_state.apply_gradients(grads=grad)

            training_state = training_state.replace(
                jepa_state=new_jepa_state
            )
            return training_state, metrics
    else:
        @jax.jit
        def update_jepa(transitions, training_state, key, offline_batch, offline_ratio):
            return training_state, {}

    @jax.jit
    def jepa_sgd_step(carry, scan_data):
        training_state, key = carry
        transitions, offline_batch, offline_ratio = scan_data
        key, jepa_key = jax.random.split(key)

        training_state, jepa_metrics = update_jepa(
            transitions, training_state, jepa_key, offline_batch, offline_ratio
        )

        return (
            training_state,
            key,
        ), jepa_metrics

    @jax.jit
    def ac_sgd_step(carry, scan_data):
        training_state, key = carry
        transitions = scan_data
        (
            key,
            critic_key,
            actor_key,
        ) = jax.random.split(key, 3)

        training_state, actor_metrics = update_actor_and_alpha(
            transitions, training_state, actor_key
        )

        training_state, critic_metrics = update_critic(
            transitions, training_state, critic_key
        )

        training_state = training_state.replace(
            gradient_steps=training_state.gradient_steps + 1
        )

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)

        return (
            training_state,
            key,
        ), metrics

    @jax.jit
    def training_step(training_state, env_state, buffer_state, key, t, offline_ratio):
        (
            experience_key1,
            experience_key2,
            sampling_key,
            training_key,
            sgd_batches_key,
            offline_key,
        ) = jax.random.split(key, 6)

        # update buffer
        env_state, buffer_state = get_experience(
            training_state,
            env_state,
            buffer_state,
            experience_key1,
        )

        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step,
        )

        transitions_list = []
        for _ in range(args.num_episodes_per_env):
            buffer_state, new_transitions = replay_buffer.sample(buffer_state)
            transitions_list.append(new_transitions)

        # Concatenate all sampled transitions
        transitions = jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=0), *transitions_list
        )

        # process transitions for training
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(
            TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0)
        )(
            (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx),
            transitions,
            batch_keys,
        )

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )

        permutation = jax.random.permutation(
            experience_key2, len(transitions.observation)
        )
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)

        # I added this code, so as to ensure len(transitions.observation) is divisible by batch_size
        num_full_batches = len(transitions.observation) // args.batch_size
        transitions = jax.tree_util.tree_map(
            lambda x: x[: num_full_batches * args.batch_size], transitions
        )

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )

        transitions_ac = transitions
        if args.use_all_batches == 0:
            num_total_batches = transitions_ac.observation.shape[0]
            selected_indices = jax.random.permutation(
                sgd_batches_key, num_total_batches
            )[: args.num_sgd_batches_per_training_step]
            transitions_ac = jax.tree_util.tree_map(
                lambda x: x[selected_indices], transitions_ac
            )

        transitions_jepa = transitions if args.jepa_use_all_batches else transitions_ac

        num_sgd_batches_jepa = transitions_jepa.observation.shape[0]
        if offline_s_t is not None:
            offline_idx = jax.random.randint(
                offline_key,
                shape=(num_sgd_batches_jepa, args.batch_size),
                minval=0,
                maxval=offline_s_t.shape[0]
            )
            offline_batches_jepa = (
                offline_s_t[offline_idx],
                offline_a_t[offline_idx],
                offline_s_tp1[offline_idx]
            )
        else:
            offline_batches_jepa = (
                jnp.zeros((num_sgd_batches_jepa, args.batch_size, args.obs_dim)),
                jnp.zeros((num_sgd_batches_jepa, args.batch_size, action_size)),
                jnp.zeros((num_sgd_batches_jepa, args.batch_size, args.obs_dim))
            )
        
        offline_ratios_jepa = jnp.full((num_sgd_batches_jepa,), offline_ratio)

        # take jepa-step worth of training-step
        (
            (
                training_state,
                training_key,
            ),
            jepa_metrics,
        ) = jax.lax.scan(jepa_sgd_step, (training_state, training_key), (transitions_jepa, offline_batches_jepa, offline_ratios_jepa))

        # take actor-step worth of training-step
        (
            (
                training_state,
                _,
            ),
            ac_metrics,
        ) = jax.lax.scan(ac_sgd_step, (training_state, training_key), transitions_ac)

        metrics = {}
        metrics.update(jepa_metrics)
        metrics.update(ac_metrics)

        return (
            training_state,
            env_state,
            buffer_state,
        ), metrics

    @jax.jit
    def training_epoch(
        training_state,
        env_state,
        buffer_state,
        key,
        offline_ratio,
    ):
        @jax.jit
        def f(carry, t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            (
                (
                    ts,
                    es,
                    bs,
                ),
                metrics,
            ) = training_step(ts, es, bs, train_key, t, offline_ratio)
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            jnp.arange(
                args.num_training_steps_per_epoch * args.training_steps_multiplier
            ),
        )

        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    key, prefill_key = jax.random.split(key, 2)

    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )

    def render_policy(training_state, save_path, epoch_num=None):
        """Renders the policy and saves it as an HTML file. Returns the html string."""

        @jax.jit
        def policy_step(env_state, actor_params, jepa_params, g_params):
            if args.jepa_actor:
                encoded_state = jax.lax.stop_gradient(jepa_state_encoder.apply(
                    jepa_params,
                    env_state.obs[: args.obs_dim]
                ))
                goal_embed = g_encoder.apply(
                    g_params,
                    env_state.obs[args.obs_dim :]
                )
                actor_input = jnp.concatenate((encoded_state, goal_embed), axis=-1)
            else:
                actor_input = env_state.obs

            means, _ = actor.apply(actor_params, actor_input)
            actions = nn.tanh(means)
            next_state = env.step(env_state, actions)
            return next_state, env_state

        rollout_states = []
        for i in range(args.num_render):
            env = make_env(args.eval_env_id)

            rng = jax.random.PRNGKey(seed=i + 1)
            env_state = jax.jit(env.reset)(rng)

            for _ in range(args.vis_length):
                jepa_params = training_state.jepa_state.params["encoder"] if args.use_jepa else None
                g_params = training_state.critic_state.params["g_encoder"] if args.use_jepa else None
                env_state, current_state = policy_step(
                    env_state, 
                    training_state.actor_state.params,
                    jepa_params,
                    g_params
                )
                rollout_states.append(current_state.pipeline_state)

        # Render and save
        html_string = html.render(env.sys, rollout_states)
        file_name = f"vis_e{epoch_num}.html" if epoch_num is not None else "vis.html"
        render_path = f"{save_path}/{file_name}"
        with open(render_path, "w") as f:
            f.write(html_string)
        return html_string

    if args.eval_actor == 0:
        """Setting up evaluator"""
        evaluator = CrlEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )

    elif args.eval_actor == 1:
        key, eval_actor_key = jax.random.split(key)
        evaluator = CrlEvaluator(
            lambda training_state, env, env_state, extra_fields: actor_step(
                training_state, env, env_state, eval_actor_key, extra_fields
            ),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )

    elif args.eval_actor > 1:
        key, eval_actor_key = jax.random.split(key)
        evaluator = CrlEvaluator(
            # Replace deterministic_actor_step with a partial function of multi_sample_actor_step
            lambda training_state,
            env,
            env_state,
            extra_fields: multi_sample_actor_step(
                training_state,
                env,
                env_state,
                eval_actor_key,
                args.eval_actor,
                extra_fields,
            ),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )

    training_walltime = 0
    print("starting training....", flush=True)
    start_time = time.time()
    for ne in range(args.num_epochs):
        t = time.time()

        if ne >= args.offline_decay_epochs:
            offline_ratio = args.offline_ratio_end
        else:
            offline_ratio = args.offline_ratio_start - (args.offline_ratio_start - args.offline_ratio_end) * (ne / args.offline_decay_epochs)

        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(
            training_state, env_state, buffer_state, epoch_key, offline_ratio
        )

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time

        sps = (
            args.env_steps_per_actor_step * args.num_training_steps_per_epoch
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/envsteps": training_state.env_steps.item(),
            **{f"training/{name}": value for name, value in metrics.items()},
        }

        metrics = evaluator.run_evaluation(training_state, metrics)

        print(
            f"epoch {ne} out of {args.num_epochs} complete. metrics: {metrics}",
            flush=True,
        )

        if args.save_buffer_every_n_epochs > 0 and ne % args.save_buffer_every_n_epochs == 0:
            print(
                    "Saving replay_buffer after epoch...",
                    flush=True,
                )
            try:
                buffer_path = f"{save_path}/replay_buffer_{ne}.pkl"
                buffer_data = {
                    "buffer_state": buffer_state,
                    "max_replay_size": args.max_replay_size,
                    "batch_size": args.batch_size,
                    "num_envs": args.num_envs,
                    "episode_length": args.episode_length,
                }
                with open(buffer_path, "wb") as f:
                    pickle.dump(buffer_data, f)
                print(f"Saved replay_buffer to {buffer_path}", flush=True)
            except Exception as e:
                print(f"Error saving replay buffer after epoch {ne}: {e}", flush=True)

        # render the policy after each epoch
        if args.capture_vis_every_n_epochs > 0 and ne % args.capture_vis_every_n_epochs == 0:
            print("Rendering policy after epoch...", flush=True)
            try:
                html_string = render_policy(training_state, save_path, epoch_num=ne)
                if args.track:
                    metrics[f"vis_e{ne}"] = wandb.Html(html_string)
            except Exception as e:
                print(f"Error rendering policy after epoch {ne}: {e}", flush=True)


        if args.checkpoint:
            if ne % 5 == 0 or ne >= args.num_epochs - 3:
                # Save current policy and critic params.
                params = (
                    training_state.alpha_state.params,
                    training_state.actor_state.params,
                    training_state.critic_state.params,
                )
                path = f"{save_path}/step_{int(training_state.env_steps)}_ep{ne}.pkl"
                save_params(path, params)
                print(f"Saved params to {path}", flush=True)

        if args.track:
            wandb.log(metrics, step=ne)

            if args.wandb_mode == "offline":
                trigger_sync()


        hours_passed = (time.time() - start_time) / 3600
        print(f"Time elapsed: {hours_passed:.3f} hours", flush=True)

    if args.checkpoint:
        # Save current policy and critic params.
        params = (
            training_state.alpha_state.params,
            training_state.actor_state.params,
            training_state.critic_state.params,
        )
        path = f"{save_path}/final.pkl"
        save_params(path, params)

    # After training is complete, render the final policy
    if args.capture_vis:
        print("Rendering final policy...", flush=True)
        try:
            html_string = render_policy(training_state, save_path)
            wandb.log({"vis": wandb.Html(html_string)})
        except Exception as e:
            print(f"Error rendering final policy: {e}", flush=True)

    # After training is complete, save the Args
    if args.checkpoint:
        with open(f"{save_path}/args.pkl", "wb") as f:
            pickle.dump(args, f)
        print(f"Saved args to {save_path}/args.pkl", flush=True)

    # After training is complete, save the replay buffer (if save_buffer is 1, this takes a lot of memory)
    if args.checkpoint:
        if args.save_buffer:
            print(
                "Saving final buffer_state and buffer data (everything needed to recreate replay_buffer)...",
                flush=True,
            )
            try:
                buffer_path = f"{save_path}/final_buffer.pkl"
                buffer_data = {
                    "buffer_state": buffer_state,
                    "max_replay_size": args.max_replay_size,
                    "batch_size": args.batch_size,
                    "num_envs": args.num_envs,
                    "episode_length": args.episode_length,
                }
                with open(buffer_path, "wb") as f:
                    pickle.dump(buffer_data, f)
                print(f"Saved replay_buffer to {buffer_path}", flush=True)
            except Exception as e:
                print(f"Error saving final replay buffer: {e}", flush=True)
