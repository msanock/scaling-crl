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


class JepaSAEncoder(nn.Module):
    jepa_encoder: nn.Module
    sa_encoder_head: nn.Module
    jepa_predictor: nn.Module = None
    jepa_action_embedder: nn.Module = None
    jepa_use_predictor_representation: bool = False
    stop_jepa_gradient: bool = True

    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        z_s = self.jepa_encoder(s)
        if self.stop_jepa_gradient:
            z_s = jax.lax.stop_gradient(z_s)

        if self.jepa_use_predictor_representation:
            a_embedded = self.jepa_action_embedder(a)
            z_s_next = self.jepa_predictor(z_s, a_embedded)
            if self.stop_jepa_gradient:
                x = jax.lax.stop_gradient(z_s_next)
            else:
                x = z_s_next
        else:
            x = jnp.concatenate([z_s, a], axis=-1)

        return self.sa_encoder_head(x)



# Copied from le-wm and translated to JAX
class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""
    knots: int = 7 # 17
    num_proj: int = 64 # 1024

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



def get_jepa_loss(args, jepa_encoder, jepa_predictor, jepa_action_embedder, sig_reg, sig_reg_params):
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
        l2_loss = jnp.mean(jnp.sum((z_tp1_pred - z_tp1_target)**2, axis=-1))
        loss = l2_loss + args.sig_reg_weight * sig_reg_loss
        return loss, {"jepa_loss": loss, "jepa_l2_loss": l2_loss, "jepa_sig_reg_loss": sig_reg_loss}

    return jepa_loss


