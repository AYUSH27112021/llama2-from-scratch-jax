import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Optional, Tuple, Any

class ModelArgs:
    def __init__(
        self, 
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        vocab_size: int = 32000,
        hidden_dim: Optional[int] = None,
        multiple_of: int = 256,
        norm_eps: float = 1e-5,
        max_seq_len: int = 2048,
        dropout: float = 0.0
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute complex sinusoidal frequencies for rotary positional embedding."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs)
    return jnp.cos(freqs), jnp.sin(freqs)

def reshape_for_broadcast(freqs_cis, x):
    """Reshape frequencies for broadcasting across tensor dimensions."""
    shape = [1] * len(x.shape)
    shape[1] = freqs_cis.shape[0]
    shape[-1] = freqs_cis.shape[-1]
    return freqs_cis.reshape(shape)

def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    """Apply rotary positional embedding to query and key tensors."""
    xq_r, xq_i = jnp.split(xq.reshape(xq.shape[:-1] + (-1, 2)), 2, axis=-1)
    xk_r, xk_i = jnp.split(xk.reshape(xk.shape[:-1] + (-1, 2)), 2, axis=-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = jnp.concatenate([xq_out_r, xq_out_i], axis=-1).reshape(xq.shape)
    xk_out = jnp.concatenate([xk_out_r, xk_out_i], axis=-1).reshape(xk.shape)

    return xq_out, xk_out

def repeat_kv(x, n_rep):
    """Repeat key/value tensors along the head dimension."""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].repeat(1, 1, 1, n_rep, 1).reshape(bs, slen, n_kv_heads * n_rep, head_dim)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    epsilon: float
    
    @nn.compact
    def __call__(self, x):
        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(variance + self.epsilon) * self.param('weight', nn.initializers.ones, (x.shape[-1],))

class Attention(nn.Module):
    config: ModelArgs
    
    @nn.compact
    def __call__(self, x, freqs_cos, freqs_sin, training=False):
        config = self.config
        n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        head_dim = config.dim // config.n_heads
        n_rep = config.n_heads // n_kv_heads

        xq = nn.Dense(features=config.n_heads * head_dim, use_bias=False)(x)
        xk = nn.Dense(features=n_kv_heads * head_dim, use_bias=False)(x)
        xv = nn.Dense(features=n_kv_heads * head_dim, use_bias=False)(x)

        xq = xq.reshape(-1, x.shape[1], config.n_heads, head_dim)
        xk = xk.reshape(-1, x.shape[1], n_kv_heads, head_dim)
        xv = xv.reshape(-1, x.shape[1], n_kv_heads, head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        xk = repeat_kv(xk, n_rep)
        xv = repeat_kv(xv, n_rep)

        # Attention computation
        attn_weights = jnp.matmul(xq, jnp.swapaxes(xk, -2, -1)) / jnp.sqrt(head_dim)
        attn_mask = jnp.triu(jnp.full((x.shape[1], x.shape[1]), -jnp.inf), k=1)
        attn_weights = jax.nn.softmax(attn_weights + attn_mask)
        
        if training and config.dropout > 0:
            attn_weights = nn.Dropout(rate=config.dropout)(attn_weights)
        
        output = jnp.matmul(attn_weights, xv)
        output = output.reshape(-1, x.shape[1], config.dim)
        output = nn.Dense(features=config.dim, use_bias=False)(output)
        
        return output

class FeedForward(nn.Module):
    config: ModelArgs
    
    @nn.compact
    def __call__(self, x, training=False):
        config = self.config
        hidden_dim = config.hidden_dim or (4 * config.dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        w1 = nn.Dense(features=hidden_dim, use_bias=False)(x)
        w3 = nn.Dense(features=hidden_dim, use_bias=False)(x)
        w2 = nn.Dense(features=config.dim, use_bias=False)

        hidden = nn.silu(w1) * w3
        output = w2(hidden)
        
        if training and config.dropout > 0:
            output = nn.Dropout(rate=config.dropout)(output)
        
        return output

class TransformerBlock(nn.Module):
    config: ModelArgs
    layer_id: int
    
    @nn.compact
    def __call__(self, x, freqs_cos, freqs_sin, training=False):
        h = x + Attention(self.config)(RMSNorm(epsilon=self.config.norm_eps)(x), freqs_cos, freqs_sin, training)
        out = h + FeedForward(self.config)(RMSNorm(epsilon=self.config.norm_eps)(h), training)
        return out

class Transformer(nn.Module):
    config: ModelArgs
    
    @nn.compact
    def __call__(self, tokens, targets=None, training=False):
        config = self.config
        
        # Token embeddings
        h = nn.Embed(num_embeddings=config.vocab_size, features=config.dim)(tokens)
        
        # Dropout
        if training and config.dropout > 0:
            h = nn.Dropout(rate=config.dropout)(h)
        
        # Precompute frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(config.dim // config.n_heads, config.max_seq_len)
        freqs_cos = freqs_cos[:tokens.shape[1]]
        freqs_sin = freqs_sin[:tokens.shape[1]]
        
        # Transformer layers
        for layer_id in range(config.n_layers):
            h = TransformerBlock(config, layer_id)(h, freqs_cos, freqs_sin, training)
        
        # Final layer norm
        h = RMSNorm(epsilon=config.norm_eps)(h)
        
        # Output layer
        logits = nn.Dense(features=config.vocab_size, use_bias=False)(h)
        
        # Loss computation
        if targets is not None:
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))
            return logits, loss
        
        return logits

def create_train_state(model, key, learning_rate):
    """Create a TrainState with AdamW optimizer."""
    tx = optax.adamw(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=model.init(key, jnp.ones((1, 10), dtype=jnp.int32))['params'], tx=tx)

@jax.jit
def train_step(state, batch):
    """Perform a single training step."""
    def loss_fn(params):
        logits, loss = state.apply_fn(
            {'params': params}, 
            batch['tokens'], 
            targets=batch['targets'], 
            training=True
        )
        return loss
    
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

@jax.jit
def generate(model, params, idx, max_new_tokens, temperature=1.0, top_k=None):
    """Generate tokens using the model."""
    def generate_step(idx, _):
        logits = model.apply({'params': params}, idx)[:, -1, :]
        
        if temperature == 0.0:
            idx_next = jnp.argmax(logits, axis=-1, keepdims=True)
        else:
            logits /= temperature
            if top_k is not None:
                v = jnp.take_along_axis(logits, jnp.argsort(logits)[-top_k:], axis=-1)
                logits = jnp.where(logits < v[0], -jnp.inf, logits)
            
            probs = jax.nn.softmax(logits)
            idx_next = jax.random.categorical(random.PRNGKey(0), probs, shape=(idx.shape[0], 1))
        
        return jnp.concatenate([idx, idx_next], axis=1), None
    
    idx, _ = jax.lax.scan(generate_step, idx, None, length=max_new_tokens)
    return idx