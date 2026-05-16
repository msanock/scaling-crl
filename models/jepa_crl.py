import flax.linen as nn
import jax.numpy as jnp

from .network_utils import lecun_unfirom, residual_block


class Jepa_encoder(nn.Module):
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


class Jepa_predictor(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0

    @nn.compact
    def __call__(self, prev_s_enc: jnp.ndarray, a: jnp.ndarray):
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        x = jnp.concatenate([prev_s_enc, a], axis=-1)
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


class SA_encoder(nn.Module):
    @nn.compact

    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
