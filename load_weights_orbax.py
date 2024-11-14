from typing import NamedTuple
import json
import jax
import jax.numpy as jnp
from pathlib import Path
import orbax.checkpoint as ocp

from llama import LlamaParams, LayerParams

def load_model_config(config_path: str) -> dict:
    """Load and parse the model configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def initialize_kv_cache(
    batch_size: int,
    num_layers: int,
    max_seq_len: int,
    n_kv_heads: int,
    head_dim: int
) -> tuple[jax.Array, jax.Array]:
    """Initialize empty KV cache arrays."""
    k_cache = jnp.zeros((num_layers, batch_size, n_kv_heads, max_seq_len, head_dim), dtype=jnp.bfloat16)
    v_cache = jnp.zeros((num_layers, batch_size, n_kv_heads, max_seq_len, head_dim), dtype=jnp.bfloat16)
    return k_cache, v_cache

def load_weights(
    checkpoint_dir: str,
    config_path: str,
    batch_size: int = 1,
    max_seq_len: int = 2048
) -> LlamaParams:
    """Load weights from Orbax checkpoint with new API."""
    # Load model configuration
    config = load_model_config(config_path)
    
    # Extract model dimensions
    dim = config['dim']
    n_layers = config['n_layers']
    n_heads = config['n_heads']
    n_kv_heads = config['n_kv_heads']
    head_dim = dim // n_heads
    
    print(f"Loading {n_layers}-layer model")
    
    def create_layer_params(layer_weights, batch_size, max_seq_len, n_kv_heads, head_dim):
        """Helper to create layer parameters from checkpoint weights."""
        k_cache, v_cache = initialize_kv_cache(
            batch_size, 1, max_seq_len, n_kv_heads, head_dim
        )
        
        return LayerParams(
            q_weight=layer_weights['attention']['wq']['weight'],
            k_weight=layer_weights['attention']['wk']['weight'],
            v_weight=layer_weights['attention']['wv']['weight'],
            o_weight=layer_weights['attention']['wo']['weight'],
            gate_weight=layer_weights['feed_forward']['w1']['weight'],
            up_weight=layer_weights['feed_forward']['w2']['weight'],
            down_weight=layer_weights['feed_forward']['w3']['weight'],
            norm_x_weight=layer_weights['attention_norm']['weight'],
            norm_z_weight=layer_weights['ffn_norm']['weight'],
            k_cache=k_cache,
            v_cache=v_cache
        )
    
    # Load checkpoint with new Orbax API
    with ocp.CheckpointManager(
        directory=checkpoint_dir,
        options=ocp.CheckpointManagerOptions()
    ) as manager:
        # Restore checkpoint with StandardRestore
        ckpt = manager.restore(0, args=ocp.args.StandardRestore())
        
        # Process layer weights
        layer_params = [
            create_layer_params(
                ckpt['layers'][str(i)],
                batch_size,
                max_seq_len,
                n_kv_heads,
                head_dim
            )
            for i in range(n_layers)
        ]
        
        # Stack layer parameters
        stacked_params = jax.tree.map(
            lambda *x: jnp.stack(x),
            *layer_params
        )
        
        # Create final parameter structure
        llama_params = LlamaParams(
            layer_params=stacked_params,
            norm_main_weight=ckpt['norm']['weight'],
            output_weight=ckpt['output']['weight'],
            cos_freq=jnp.array([0]), # Placeholder for now
            sin_freq=jnp.array([0]) # Placeholder for now
        )
        
        print("Successfully loaded all weights")
        return llama_params

