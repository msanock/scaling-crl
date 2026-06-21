import flax.linen as nn
from flax.linen.initializers import variance_scaling

lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
bias_init = nn.initializers.zeros


def residual_block(x, width, normalize, activation):
    identity = x
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = x + identity
    return x


class AttentionBlock(nn.Module):
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x, x_pos_embed=0.0):
        # x shape: (batch_size, seq_len, embed_dim)
        x_with_pos = x + x_pos_embed

        # Self-Attention
        attn_out = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(
            inputs_q=x_with_pos, inputs_k=x_with_pos, inputs_v=x
        )
        x = x + attn_out
        x = nn.LayerNorm()(x)

        # MLP
        mlp_out = nn.Dense(self.mlp_dim)(x)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(x.shape[-1])(mlp_out)

        x = x + mlp_out
        x = nn.LayerNorm()(x)

        return x
