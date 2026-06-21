import jax
import flax.linen as nn
import jax.numpy as jnp

from .network_utils import lecun_unfirom, residual_block


class SA_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    output_dim: int = 64

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        x = jnp.concatenate([s, a], axis=-1)
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
        x = nn.Dense(self.output_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class G_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    output_dim: int = 64

    @nn.compact
    def __call__(self, g: jnp.ndarray):
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        x = g
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
        x = nn.Dense(self.output_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


def get_classic_critic_loss(args, sa_encoder, g_encoder):

    def critic_loss(critic_params, transitions, key):
            sa_encoder_params, g_encoder_params = (
                critic_params["sa_encoder"],
                critic_params["g_encoder"],
            )

            obs = transitions.observation[:, : args.obs_dim]
            action = transitions.action

            sa_repr = sa_encoder.apply(sa_encoder_params, obs, action)
            g_repr = g_encoder.apply(
                g_encoder_params, transitions.observation[:, args.obs_dim :]
            )

            # InfoNCE
            logits = -jnp.sqrt(
                jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)
            )  # shape = BxB
            critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

            # logsumexp regularisation
            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            critic_loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

            I, correct, logits_pos, logits_neg = (
                jnp.zeros(1),
                jnp.zeros(1),
                jnp.zeros(1),
                jnp.zeros(1),
            )

            return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg)

    return critic_loss