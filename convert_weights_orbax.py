import os
import torch
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import argparse
import gc
from typing import Dict, Any
import orbax.checkpoint as ocp

def setup_args():
    parser = argparse.ArgumentParser(description='Convert LLaMA weights from PyTorch to JAX using Orbax')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Path to PyTorch weights file')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save Orbax checkpoint')
    return parser.parse_args()

def convert_tensor_to_jax(tensor: torch.Tensor) -> jax.Array:
    """Convert PyTorch tensor to JAX array in bfloat16."""
    # Convert to float32 first
    tensor_f32 = tensor.to(torch.float32)
    del tensor
    gc.collect()
    
    # Convert to numpy
    numpy_tensor = tensor_f32.detach().cpu().numpy()
    del tensor_f32
    gc.collect()
    
    # Convert to JAX and bfloat16
    jax_array = jnp.array(numpy_tensor, dtype=jnp.bfloat16)
    del numpy_tensor
    gc.collect()
    
    return jax_array

class LazyLoadStateDictIterator:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        loaded = torch.load(
            self.checkpoint_path,
            map_location='cpu',
            weights_only=True
        )
        if hasattr(loaded, 'state_dict'):
            loaded = loaded.state_dict()
        self.keys = list(loaded.keys())
        del loaded
        gc.collect()

    def __iter__(self):
        for key in self.keys:
            def partial_load(storage, location):
                return storage

            tensor = torch.load(
                self.checkpoint_path,
                map_location=partial_load,
                weights_only=True
            )[key]
            
            yield key, tensor
            
            del tensor
            gc.collect()
    
    def __len__(self):
        return len(self.keys)

def convert_weights(input_path: str, output_dir: str) -> None:
    """Convert weights and save as Orbax checkpoint."""
    # Initialize Orbax checkpointer with PyTree handler
    options = ocp.CheckpointManagerOptions(create=True, max_to_keep=1)

    # Use context manager pattern with new API
    with ocp.CheckpointManager(
        directory=output_dir,
        options=options
    ) as manager:
        # Create iterator for lazy loading
        iterator = LazyLoadStateDictIterator(input_path)
        total_params = len(iterator)
        
        # Dictionary to store converted weights
        converted_weights = {}
        
        # Process each parameter
        for i, (param_name, param_tensor) in enumerate(iterator, 1):
            print(f"Converting {param_name} ({i}/{total_params})")
            
            # Convert to JAX array
            jax_array = convert_tensor_to_jax(param_tensor)
            
            # Store in dictionary with proper structure
            name_parts = param_name.split('.')
            current_dict = converted_weights
            for part in name_parts[:-1]:
                current_dict = current_dict.setdefault(part, {})
            current_dict[name_parts[-1]] = jax_array
            
            print(f"Completed converting {param_name}")
        
        # Save checkpoint
        print("Saving checkpoint...")
        manager.save(0, args=ocp.args.StandardSave(converted_weights))
        manager.wait_until_finished()
        print("Checkpoint saved successfully")

def main():
    args = setup_args()
    print(f"Starting conversion from {args.input_dir} to {args.output_dir}")
    convert_weights(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()