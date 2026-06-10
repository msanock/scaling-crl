import flax.linen as nn
import jax.numpy as jnp
import jax

from .network_utils import lecun_unfirom, residual_block

class JepaEncoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0

    @nn.compact
    def __call__(self, s: jnp.ndarray):
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        x = s
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
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class JepaActionEmbedder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0

    @nn.compact
    def __call__(self, a: jnp.ndarray):
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        x = a
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
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x





class JepaPredictor(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0

    @nn.compact
    def __call__(self, prev_s_enc: jnp.ndarray, a_embedding: jnp.ndarray):
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        # TODO: verify what action should be done here
        x = jnp.concatenate([prev_s_enc, a_embedding], axis=-1)
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
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x

class JepaIDM(nn.Module):
    """Inverse Dynamics Model: predicts a_t from (z_t, z_{t+1}).

    L_IDM = ||a_t - IDM(z_t, z_{t+1})||^2  (Pathak et al. 2017)
    Grounding consecutive embeddings in action space prevents the encoder
    from collapsing or developing spurious correlations.
    """
    action_size: int
    network_width: int = 128
    use_relu: int = 0

    @nn.compact
    def __call__(self, z_t: jnp.ndarray, z_tp1: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([z_t, z_tp1], axis=-1)  # (B, 2*D)
        activation = nn.relu if self.use_relu else nn.swish
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom)(x)
        x = activation(x)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom)(x)
        x = activation(x)
        x = nn.Dense(self.action_size, kernel_init=lecun_unfirom)(x)
        return x  # predicted action, NOT tanh-squashed (raw regression target)


class SA_encoderHead(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0

    @nn.compact
    def __call__(self, x):
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        # Initial layer
        x = nn.Dense(
            self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init
        )(x)
        x = normalize(x)
        x = activation(x)
        # Residual blocks
        for i in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x






# Copied from le-wm and translated to JAX
class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""
    knots: int = 17 # 17
    num_proj: int = 1024 # 1024

    def setup(self):
        t = jnp.linspace(0, 3, self.knots, dtype=jnp.float32)
        dt = 3 / (self.knots - 1)
        weights = jnp.full((self.knots,), 2 * dt, dtype=jnp.float32)
        weights = weights.at[0].set(dt)
        weights = weights.at[-1].set(dt)
        window = jnp.exp(-jnp.square(t) / 2.0)
        self.t = t
        self.phi = window
        self.weights = weights * window

    def __call__(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = jax.random.normal(self.make_rng("proj"), (proj.shape[-1], self.num_proj))
        A = A / jnp.linalg.norm(A, axis=0, keepdims=True)
        # compute the epps-pulley statistic
        x_t = jnp.expand_dims(proj @ A, axis=-1) * self.t
        err = jnp.square(jnp.mean(jnp.cos(x_t), axis=-3) - self.phi) + jnp.square(jnp.mean(jnp.sin(x_t), axis=-3))
        statistic = (err @ self.weights) * proj.shape[-2]
        return statistic.mean() # average over projections and time



def vicreg_loss(z: jnp.ndarray, gamma: float = 1.0, epsilon: float = 1e-4):
    """VICReg variance + covariance regularisation.

    Variance term: penalises per-dimension std < gamma, preventing collapse.
    Covariance term: penalises off-diagonal of the feature covariance matrix,
    encouraging each dimension to capture an independent factor of variation
    (i.e. structural sparsity / concentrated PCA variance).

    Args:
        z: (B, D) encoder embeddings.
        gamma: target minimum std per dimension (default 1.0).
        epsilon: numerical stability for std computation.
    Returns:
        (var_loss, cov_loss) scalars.
    """
    B, D = z.shape
    z = z - z.mean(axis=0, keepdims=True)  # centre
    # Variance term
    std = jnp.sqrt(z.var(axis=0) + epsilon)
    var_loss = jnp.mean(jax.nn.relu(gamma - std))
    # Covariance term
    cov = (z.T @ z) / (B - 1)  # (D, D)
    off_diag = cov - jnp.diag(jnp.diag(cov))
    cov_loss = jnp.sum(jnp.square(off_diag)) / D
    return var_loss, cov_loss


def sim_loss(z_seq: jnp.ndarray) -> jnp.ndarray:
    """Temporal similarity loss: encourages smooth latent trajectories.

    L_sim = Σ_t ||z_t - z_{t+1}||^2  (averaged over t and batch).

    Args:
        z_seq: (T, B, D) sequence of embeddings (T >= 2).
    Returns:
        Scalar loss.
    """
    return jnp.mean(jnp.sum((z_seq[1:] - z_seq[:-1]) ** 2, axis=-1))


def get_jepa_loss(
    args,
    jepa_encoder,
    jepa_predictor,
    jepa_action_embedder,
    sig_reg,
    sig_reg_params,
    jepa_idm=None,
    jepa_idm_params=None,
):
    def jepa_loss(jepa_params, transitions, key):
        s_t = transitions.extras["state"]
        a_t = transitions.action
        s_tp1 = transitions.extras["next_state"]

        proj_key = jax.random.split(key, 2)[1]

        encoder_params = jepa_params["encoder"]
        predictor_params = jepa_params["predictor"]
        action_embedder_params = jepa_params["action_embedder"]

        z_t = jepa_encoder.apply(encoder_params, s_t)
        a_emb = jepa_action_embedder.apply(action_embedder_params, a_t)
        z_tp1_pred = jepa_predictor.apply(predictor_params, z_t, a_emb)
        z_tp1_target = jepa_encoder.apply(encoder_params, s_tp1)

        sig_reg_loss = sig_reg.apply(sig_reg_params, z_t, rngs={"proj": proj_key})
        l2_loss = jnp.mean(jnp.sum((z_tp1_pred - z_tp1_target) ** 2, axis=-1))

        var_loss_weight = getattr(args, "var_loss_weight", 0.0)
        cov_loss_weight = getattr(args, "cov_loss_weight", 0.0)
        sim_loss_weight = getattr(args, "sim_loss_weight", 0.0)
        idm_loss_weight = getattr(args, "idm_loss_weight", 0.0)

        var_loss, cov_loss = vicreg_loss(z_t)

        # Temporal similarity loss (single-step: just two embeddings)
        z_seq_1step = jnp.stack([z_t, z_tp1_target], axis=0)  # (2, B, D)
        sim_loss_val = sim_loss(z_seq_1step)

        # IDM loss
        if jepa_idm is not None and jepa_idm_params is not None and idm_loss_weight > 0.0:
            a_pred = jepa_idm.apply(jepa_idm_params, z_t, z_tp1_target)
            idm_loss_val = jnp.mean(jnp.sum((a_t - a_pred) ** 2, axis=-1))
        else:
            idm_loss_val = jnp.zeros(())

        loss = (
            l2_loss
            + args.sig_reg_weight * sig_reg_loss
            + var_loss_weight * var_loss
            + cov_loss_weight * cov_loss
            + sim_loss_weight * sim_loss_val
            + idm_loss_weight * idm_loss_val
        )
        return loss, {
            "jepa_loss": loss,
            "jepa_l2_loss": l2_loss,
            "jepa_sig_reg_loss": sig_reg_loss,
            "jepa_var_loss": var_loss,
            "jepa_cov_loss": cov_loss,
            "jepa_sim_loss": sim_loss_val,
            "jepa_idm_loss": idm_loss_val,
        }

    return jepa_loss


def get_multistep_jepa_loss(
    args,
    jepa_encoder,
    jepa_predictor,
    jepa_action_embedder,
    sig_reg,
    sig_reg_params,
    jepa_idm=None,
    jepa_idm_params=None,
):
    """Multi-step rollout loss for JEPA.

    Expects windowed batches with shape:
        states:  (B, K+1, D_obs)  — K+1 consecutive observations
        actions: (B, K, D_act)    — K consecutive actions

    Unrolls the predictor K steps propagating *predicted* embeddings forward:
        L_pred = (1/K) Σ_{k=1..K} ||z_{t+k}^pred − z_{t+k}^target||^2
        L_sim  = smooth-trajectory regulariser over all K+1 *target* embeddings
        L_IDM  = (1/K) Σ_{k=0..K-1} ||a_{t+k} − IDM(z_{t+k}, z_{t+k+1})||^2
    """
    def jepa_loss(jepa_params, states, actions, key):
        """
        states:  (B, K+1, D_obs)
        actions: (B, K, D_act)
        """
        proj_key = jax.random.split(key, 2)[1]
        B, Kp1, D_obs = states.shape
        K = actions.shape[1]

        encoder_params = jepa_params["encoder"]
        predictor_params = jepa_params["predictor"]
        action_embedder_params = jepa_params["action_embedder"]

        # Encode all K+1 states: reshape → encode → reshape back
        z_targets = jepa_encoder.apply(
            encoder_params, states.reshape(B * Kp1, D_obs)
        ).reshape(B, Kp1, -1)  # (B, K+1, D_emb)
        z_targets = jax.lax.stop_gradient(z_targets)
        D_emb = z_targets.shape[-1]

        # Multi-step rollout (Python loop, unrolled at trace time)
        pred_losses = []
        z_current = z_targets[:, 0, :]  # z_t, shape (B, D_emb)
        for k in range(K):
            a_k = actions[:, k, :]
            a_emb_k = jepa_action_embedder.apply(action_embedder_params, a_k)
            z_next_pred = jepa_predictor.apply(predictor_params, z_current, a_emb_k)
            step_loss = jnp.mean(jnp.sum((z_next_pred - z_targets[:, k + 1, :]) ** 2, axis=-1))
            pred_losses.append(step_loss)
            z_current = z_next_pred  # propagate predicted embedding

        l2_loss = jnp.mean(jnp.stack(pred_losses))

        # SIGReg on z_t only
        sig_reg_loss = sig_reg.apply(
            sig_reg_params, z_targets[:, 0, :], rngs={"proj": proj_key}
        )

        # VICReg on all target embeddings flattened over time
        z_flat = z_targets.reshape(B * Kp1, D_emb)
        var_loss_weight = getattr(args, "var_loss_weight", 0.0)
        cov_loss_weight = getattr(args, "cov_loss_weight", 0.0)
        var_loss, cov_loss = vicreg_loss(z_flat)

        # Temporal similarity loss over target embedding sequence
        z_seq = jnp.transpose(z_targets, (1, 0, 2))  # (K+1, B, D_emb)
        sim_loss_weight = getattr(args, "sim_loss_weight", 0.0)
        sim_loss_val = sim_loss(z_seq)

        # IDM loss over consecutive target pairs
        idm_loss_weight = getattr(args, "idm_loss_weight", 0.0)
        if jepa_idm is not None and jepa_idm_params is not None and idm_loss_weight > 0.0:
            idm_step_losses = []
            for k in range(K):
                z_k   = z_targets[:, k, :]
                z_kp1 = z_targets[:, k + 1, :]
                a_k   = actions[:, k, :]
                a_pred = jepa_idm.apply(jepa_idm_params, z_k, z_kp1)
                idm_step_losses.append(jnp.mean(jnp.sum((a_k - a_pred) ** 2, axis=-1)))
            idm_loss_val = jnp.mean(jnp.stack(idm_step_losses))
        else:
            idm_loss_val = jnp.zeros(())

        loss = (
            l2_loss
            + args.sig_reg_weight * sig_reg_loss
            + var_loss_weight * var_loss
            + cov_loss_weight * cov_loss
            + sim_loss_weight * sim_loss_val
            + idm_loss_weight * idm_loss_val
        )
        return loss, {
            "jepa_loss": loss,
            "jepa_l2_loss": l2_loss,
            "jepa_sig_reg_loss": sig_reg_loss,
            "jepa_var_loss": var_loss,
            "jepa_cov_loss": cov_loss,
            "jepa_sim_loss": sim_loss_val,
            "jepa_idm_loss": idm_loss_val,
        }

    return jepa_loss
