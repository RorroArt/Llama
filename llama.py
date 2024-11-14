import jax
import jax.numpy as jnp

from functools import partial
import einops

from typing import NamedTuple, Callable, Tuple

def rms_norm(x, weight, eps=1e-5):
    mean_square = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_x = x * jax.lax.rsqrt(mean_square + eps)
    return normed_x * weight

def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    xq = xq.astype(jnp.float32)
    xk = xk.astype(jnp.float32)
    xqri = einops.rearrange(xq, 'b l h (d r) -> b l h d r', r=2)
    xkri = einops.rearrange(xk, 'b l h (d r) -> b l h d r', r=2)

    # Reshape `xq` and `xk` to match the complex representation.
    xq_r, xq_i = jnp.split(xqri, 2, axis=-1)
    xq_r = xq_r.squeeze(-1)
    xq_i = xq_i.squeeze(-1)

    xk_r, xk_i = jnp.split(xkri, 2, axis=-1)
    xk_r = xk_r.squeeze(-1)
    xk_i = xk_i.squeeze(-1)

    # Reshape `freqs_cos` and `freqs_sin` for broadcasting.
    freqs_cos = partial(einops.rearrange, pattern='l d -> 1 l 1 d')(freqs_cos)
    freqs_sin = partial(einops.rearrange, pattern='l d -> 1 l 1 d')(freqs_sin)

    # Apply rotation using real numbers.
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Flatten last two dimensions.
    xq_out = jnp.stack([xq_out_r, xq_out_i], axis=-1)
    xk_out = jnp.stack([xk_out_r, xk_out_i], axis=-1)
    xq_out = partial(einops.rearrange, pattern='b l h d r -> b l h (d r)')(xq_out)
    xk_out = partial(einops.rearrange, pattern='b l h d r -> b l h (d r)')(xk_out)
    
    xq_out = xq_out.astype(jnp.bfloat16)
    xk_out = xk_out.astype(jnp.bfloat16)
    return xq_out, xk_out

def ffn(x, gate_weight, up_weight, down_weight):
    B, L, D = x.shape
    FD = gate_weight.shape[-1]  # FD = 4 * D

    # Compute the gating function
    gate = jax.nn.silu(einops.einsum(x, gate_weight, 'b l d, f d -> b l f'))  # Shape: (B, L, FD)

    # Compute the up projection
    up = einops.einsum(x, up_weight, 'b l d, f d -> b l f')  # Shape: (B, L, FD)

    # Element-wise multiplication
    x = gate * up

    # Compute the down projection
    x = einops.einsum(x, down_weight, 'b l f, d f -> b l d')  # Shape: (B, L, D)

    return x

class KVCache(NamedTuple):
    k: jax.Array
    v: jax.Array

def update_kv_cache(cache: KVCache, xk: jax.Array, xv: jax.Array, start_pos: int) -> KVCache:
    """Update KV cache without using position masks."""
    k_cache = jax.lax.dynamic_update_slice_in_dim(cache.k, xk, start_pos, axis=1)
    v_cache = jax.lax.dynamic_update_slice_in_dim(cache.v, xv, start_pos, axis=1)
    return KVCache(k_cache, v_cache)

def attention(x: jax.Array, 
             cache: KVCache, 
             start_pos: int, 
             freqs_cos: jax.Array, 
             freqs_sin: jax.Array, 
             q_weight: jax.Array, 
             k_weight: jax.Array, 
             v_weight: jax.Array, 
             o_weight: jax.Array, 
             mask: jax.Array = None) -> Tuple[jax.Array, KVCache]:
    """Optimized attention implementation."""
    B, L, D = x.shape
    _, S, KVH, K = cache.k.shape
    H = q_weight.shape[1] // K

    # QKV projections with optimized einsum
    xq = jnp.einsum('bld,dk->blk', x, q_weight)
    xk = jnp.einsum('bld,dk->blk', x, k_weight)
    xv = jnp.einsum('bld,dk->blk', x, v_weight)

    # Reshape for multi-head attention
    xq = einops.rearrange(xq, 'b l (h k) -> b l h k', h=H, k=K)
    xk = einops.rearrange(xk, 'b l (h k) -> b l h k', h=KVH, k=K)
    xv = einops.rearrange(xv, 'b l (h k) -> b l h k', h=KVH, k=K)
    
    # Apply rotary embeddings
    xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
    
    # Update cache
    new_cache = update_kv_cache(cache, xk, xv, start_pos)
    
    # Extract relevant key-value pairs using dynamic slice
    # This replaces the position mask approach
    relevant_length = start_pos + L
    k_relevant = jax.lax.dynamic_slice(
        new_cache.k,
        (0, 0, 0, 0),
        (B, relevant_length, KVH, K)
    )
    v_relevant = jax.lax.dynamic_slice(
        new_cache.v,
        (0, 0, 0, 0),
        (B, relevant_length, KVH, K)
    )

    # Handle head repetition for MQA/GQA
    n_rep = H // KVH
    if n_rep > 1:
        k_relevant = jnp.repeat(k_relevant, n_rep, axis=2)
        v_relevant = jnp.repeat(v_relevant, n_rep, axis=2)

    # Reshape for attention computation
    xq = einops.rearrange(xq, 'b l h k -> b h l k')
    k_relevant = einops.rearrange(k_relevant, 'b l h k -> b h l k')
    v_relevant = einops.rearrange(v_relevant, 'b l h k -> b h l k')

    # Compute attention scores with fused scaling
    scale = 1.0 / jnp.sqrt(K)
    attention = jnp.einsum('bhik,bhjk->bhij', xq, k_relevant) * scale

    # Apply attention mask if provided
    if mask is not None:
        attention = attention + mask[None, None, :, :relevant_length]

    # Compute attention weights and output
    attention = jax.nn.softmax(attention, axis=-1)
    output = jnp.einsum('bhij,bhjk->bhik', attention, v_relevant)

    # Final projection
    output = einops.rearrange(output, 'b h l k -> b l (h k)')
    output = jnp.einsum('bld,dh->blh', output, o_weight)

    return output, new_cache

def rms_norm(x, weight, eps=1e-6):
    mean_square = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_x = x * jax.lax.rsqrt(mean_square + eps)
    return normed_x * weight


def transformer(x, start, cos_freq, sin_freq, params, mask=None):
    cache = KVCache(k=params.k_cache, v=params.v_cache)

    norm_x = rms_norm(x, params.norm_x_weight)
    h1, new_cache = attention(norm_x, cache, start, cos_freq, sin_freq, params.q_weight, params.k_weight, params.v_weight, params.o_weight, mask)
    z = x + h1
    norm_z = rms_norm(z, params.norm_z_weight)
    h2 = ffn(norm_z, params.gate_weight, params.up_weight, params.down_weight)
    out = z + h2

    return out, new_cache


def create_full_mask(L, start, max_len):
    full_mask = jnp.full((L, max_len), float("-inf"))

    tri_mask = jnp.triu(jnp.full((L, L), float("-inf")), 1)

    full_mask = jax.lax.dynamic_update_slice_in_dim(full_mask, tri_mask, start, axis=1)

    full_mask = jnp.where(jnp.arange(max_len)[None, :] < start, 0, full_mask)

    return full_mask

class LayerParams(NamedTuple):
    q_weight: jax.Array
    k_weight: jax.Array
    v_weight: jax.Array
    o_weight: jax.Array
    gate_weight: jax.Array
    up_weight: jax.Array
    down_weight:jax.Array
    norm_x_weight: jax.Array
    norm_z_weight: jax.Array
    k_cache: jax.Array
    v_cache: jax.Array

class LlamaParams(NamedTuple):
    layer_params: LayerParams
    norm_main_weight: jax.Array
    output_weight: jax.Array
    cos_freq: jax.Array
    sin_freq: jax.Array
    

# @partial(jax.jit, static_argnames=['start'])
@jax.jit
def llama(x, start, params, max_seq_len=5000):
    B = x.shape[0]
    L = x.shape[1]
    

    cos_freq = jax.lax.dynamic_slice_in_dim(params.cos_freq, start, L, axis=0)
    sin_freq = jax.lax.dynamic_slice_in_dim(params.sin_freq, start, L, axis=0)
    # cos_freq = params.cos_freq[start:start+L]
    # sin_freq = params.sin_freq[start:start+L]
    mask = create_full_mask(L, start, max_seq_len).astype(jnp.bfloat16)

    def layer_fn(start, cos_freq, sin_freq, mask, x, layer_weights):
        x, new_cache = transformer(x, start, cos_freq, sin_freq, layer_weights, mask)
        return x, (new_cache.k, new_cache.v)

    scan_fn = partial(layer_fn, start, cos_freq, sin_freq, mask)

    x, new_cache = jax.lax.scan(scan_fn, x, params.layer_params)

    h = rms_norm(x, params.norm_main_weight)
    logit = jnp.dot(h[:, [-1],:], params.output_weight)
    return logit, new_cache

def top_p_sampling(probs: jnp.ndarray, p: float, key: jax.random.PRNGKey) -> jnp.ndarray:
    """Perform top-p (nucleus) sampling on a probability distribution."""
    sorted_indices = jnp.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = jnp.cumsum(sorted_probs)
    cutoff = jnp.argmax(cumulative_probs > p)
    top_p_indices = sorted_indices[:cutoff + 1]
    top_p_probs = sorted_probs[:cutoff + 1]
    top_p_probs /= jnp.sum(top_p_probs)
    return jax.random.choice(key, top_p_indices, p=top_p_probs)

def generate(
    params: LlamaParams,
    config: dict,
    input_ids: jnp.ndarray,
    max_new_tokens: int,
    tokenizer_model: jnp.ndarray,
    model_fn: Callable,
    temperature: float = 0.3,
    top_p: float = 0.9,
    seed: int = 2
):
    """Generate tokens using the model with top-p sampling."""
    max_seq_len = config['max_seq_len']
    key = jax.random.PRNGKey(seed)
    og_len = input_ids.shape[1]
    

    pos = 0 
    while pos < max_new_tokens:
        if pos == 0:
            inputs = input_ids
            curr_pos = 0
        else:
            inputs = input_ids[:, -1:]
            curr_pos = pos + og_len - 1

        h = tokenizer_model[inputs]

        logits, new_cache = model_fn(h, curr_pos, params)
        

        if temperature > 0:
            probs = jax.nn.softmax(logits[:, -1] / temperature, axis=-1)
            next_key, key = jax.random.split(key)
            next_token = top_p_sampling(probs[0], top_p, next_key)
        else:
            next_token = logits[:, -1, :].argmax(-1).squeeze(0)

        params = params._replace(
            layer_params=params.layer_params._replace(
                k_cache=new_cache[0],
                v_cache=new_cache[1]
            )
        )

        input_ids = jnp.concatenate([input_ids, next_token[None, None]], axis=1)
        pos += 1

    return input_ids[:, og_len:], params
