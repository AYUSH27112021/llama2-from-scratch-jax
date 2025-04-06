import os
import struct
import gzip
import shutil
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

# -----------------------------------------------------------------------------
# common utilities

def serialize_fp32(file, array):
    """ writes one fp32 array to file that is open in wb mode """
    d = np.asarray(array).astype(np.float32).flatten()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def serialize_int8(file, array):
    """ writes one int8 array to file that is open in wb mode """
    d = np.asarray(array).astype(np.int8).flatten()
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)

def quantize_q80(w, group_size):
    """
    takes a JAX array and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.size % group_size == 0
    ori_shape = w.shape
    w = w.astype(jnp.float32)
    w = w.reshape(-1, group_size)
    wmax = jnp.max(jnp.abs(w), axis=1)
    scale = wmax / 127.0
    quant = w / scale[:, None]
    int8val = jnp.round(quant).astype(jnp.int8)
    fp32val = (int8val.astype(jnp.float32) * scale[:, None]).flatten()
    fp32valr = fp32val.reshape(-1, group_size)
    err = jnp.max(jnp.abs(fp32valr - w), axis=1)
    maxerr = jnp.max(err).item()
    return int8val, scale, maxerr

# -----------------------------------------------------------------------------
# legacy export (v0)

def legacy_export(model, params, filepath):
    """ Original export of llama2.c bin files, i.e. version v0 """
    out_file = open(filepath, 'wb')

    flat_params = flatten_dict(params)
    
    model_config = model.config
    
    # Check for shared embeddings in JAX model
    tok_embeddings = flat_params.get(('tok_embeddings', 'embedding'))
    output_weights = flat_params.get(('output', 'kernel'))
    shared_classifier = jnp.array_equal(tok_embeddings, output_weights.T) if tok_embeddings is not None and output_weights is not None else False
    
    # Get hidden dimension
    hidden_dim = flat_params[('layers', '0', 'feed_forward', 'w1', 'kernel')].shape[0]
    
    # first write out the header
    p = model_config
    # legacy format uses negative/positive vocab_size as a shared classifier flag
    vocab_size = p.vocab_size if shared_classifier else -p.vocab_size
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, vocab_size, p.max_seq_len)
    out_file.write(header)

    # next write out the embedding weights
    serialize_fp32(out_file, tok_embeddings)

    # attention weights
    for i in range(p.n_layers):
        serialize_fp32(out_file, flat_params[('layers', f'{i}', 'attention_norm', 'scale')])
    for i in range(p.n_layers):
        serialize_fp32(out_file, flat_params[('layers', f'{i}', 'attention', 'wq', 'kernel')].T)
    for i in range(p.n_layers):
        serialize_fp32(out_file, flat_params[('layers', f'{i}', 'attention', 'wk', 'kernel')].T)
    for i in range(p.n_layers):
        serialize_fp32(out_file, flat_params[('layers', f'{i}', 'attention', 'wv', 'kernel')].T)
    for i in range(p.n_layers):
        serialize_fp32(out_file, flat_params[('layers', f'{i}', 'attention', 'wo', 'kernel')].T)
    # ffn weights
    for i in range(p.n_layers):
        serialize_fp32(out_file, flat_params[('layers', f'{i}', 'ffn_norm', 'scale')])
    for i in range(p.n_layers):
        serialize_fp32(out_file, flat_params[('layers', f'{i}', 'feed_forward', 'w1', 'kernel')].T)
    for i in range(p.n_layers):
        serialize_fp32(out_file, flat_params[('layers', f'{i}', 'feed_forward', 'w2', 'kernel')].T)
    for i in range(p.n_layers):
        serialize_fp32(out_file, flat_params[('layers', f'{i}', 'feed_forward', 'w3', 'kernel')].T)
    # final rmsnorm
    serialize_fp32(out_file, flat_params[('norm', 'scale')])
    # freqs_cis
    if ('freqs_cos',) in flat_params and ('freqs_sin',) in flat_params:
        serialize_fp32(out_file, flat_params[('freqs_cos',)][:p.max_seq_len])
        serialize_fp32(out_file, flat_params[('freqs_sin',)][:p.max_seq_len])
    else:
        # Generate the frequency data if not in the model parameters
        theta = 10000.0
        freqs = 1.0 / (theta ** (jnp.arange(0, p.dim // 2, 2).astype(jnp.float32) / p.dim))
        t = jnp.arange(p.max_seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, freqs)
        freqs_cos = jnp.cos(freqs)
        freqs_sin = jnp.sin(freqs)
        serialize_fp32(out_file, freqs_cos)
        serialize_fp32(out_file, freqs_sin)

    # final classifier weights
    if not shared_classifier:
        serialize_fp32(out_file, output_weights.T)

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")

# -----------------------------------------------------------------------------
# (v1) --- implementation

def version1_export(model, params, filepath):
    """
    Export the model weights in full float32 .bin file to be read from C.
    This is same as legacy_export, but with a proper header.
    """
    version = 1

    out_file = open(filepath, 'wb')

    flat_params = flatten_dict(params)
    model_config = model.config
    tok_embeddings = flat_params.get(('tok_embeddings', 'embedding'))
    output_weights = flat_params.get(('output', 'kernel'))
    shared_classifier = jnp.array_equal(tok_embeddings, output_weights.T) if tok_embeddings is not None and output_weights is not None else False
    hidden_dim = flat_params[('layers', '0', 'feed_forward', 'w1', 'kernel')].shape[0]
    # the header will be 256 bytes
    # 1) write magic, which will be uint32 of "ak42" in ASCII
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) write version, which will be int
    out_file.write(struct.pack('i', version))
    # 3) write the params, which will be 7 ints
    p = model_config
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) write some other flags
    out_file.write(struct.pack('B', int(shared_classifier)))
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)

    weights = []
    # attention norm
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'attention_norm', 'scale')])
    # ffn norm
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'ffn_norm', 'scale')])
    # final norm
    weights.append(flat_params[('norm', 'scale')])
    # embedding
    weights.append(tok_embeddings)
    # attention wq
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'attention', 'wq', 'kernel')].T)
    # attention wk
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'attention', 'wk', 'kernel')].T)
    # attention wv
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'attention', 'wv', 'kernel')].T)
    # attention wo
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'attention', 'wo', 'kernel')].T)
    # feed forward w1
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'feed_forward', 'w1', 'kernel')].T)
    # feed forward w2
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'feed_forward', 'w2', 'kernel')].T)
    # feed forward w3
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'feed_forward', 'w3', 'kernel')].T)
    
    if not shared_classifier:
        weights.append(output_weights.T)
    
    for w in weights:
        serialize_fp32(out_file, w)

    out_file.close()
    print(f"wrote {filepath}")

# -----------------------------------------------------------------------------
# v2 --- implementation - quantized

def version2_export(model, params, filepath, group_size=64):
    """
    Export the model weights in Q8_0 into .bin file to be read from C.
    That is:
    - quantize all weights to symmetric int8, in range [-127, 127]
    - all other tensors (the rmsnorm params) are kept and exported in fp32
    - quantization is done in groups of group_size to reduce the effects of any outliers
    """
    version = 2
    flat_params = flatten_dict(params)
    model_config = model.config
    hidden_dim = flat_params[('layers', '0', 'feed_forward', 'w1', 'kernel')].shape[0]
    p = model_config
    while p.dim % group_size != 0:
        group_size //= 2
        print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")
    tok_embeddings = flat_params.get(('tok_embeddings', 'embedding'))
    output_weights = flat_params.get(('output', 'kernel'))
    shared_classifier = jnp.array_equal(tok_embeddings, output_weights.T) if tok_embeddings is not None and output_weights is not None else False
    weights = [
        tok_embeddings
    ]
    # attention
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'attention', 'wq', 'kernel')].T)
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'attention', 'wk', 'kernel')].T)
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'attention', 'wv', 'kernel')].T)
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'attention', 'wo', 'kernel')].T)
    # feed forward
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'feed_forward', 'w1', 'kernel')].T)
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'feed_forward', 'w2', 'kernel')].T)
    for i in range(p.n_layers):
        weights.append(flat_params[('layers', f'{i}', 'feed_forward', 'w3', 'kernel')].T)
    
    if not shared_classifier:
        weights.append(output_weights.T)
    
    # validate
    for w in weights:
        assert w.size % group_size == 0, f"weight has size {w.size}, not a multiple of group_size {group_size}"
    # write
    out_file = open(filepath, 'wb')
    out_file.write(struct.pack('I', 0x616b3432))
    out_file.write(struct.pack('i', version))
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    out_file.write(struct.pack('B', int(shared_classifier)))
    out_file.write(struct.pack('i', group_size))
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)
    
    for i in range(p.n_layers):
        serialize_fp32(out_file, flat_params[('layers', f'{i}', 'attention_norm', 'scale')])
    for i in range(p.n_layers):
        serialize_fp32(out_file, flat_params[('layers', f'{i}', 'ffn_norm', 'scale')])
    serialize_fp32(out_file, flat_params[('norm', 'scale')])

    ew = []
    for i, w in enumerate(weights):
        q, s, err = quantize_q80(w, group_size)
        # int8
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
        # logging
        ew.append((err, w.shape))
        print(f"{i+1}/{len(weights)} quantized {tuple(w.shape)} to Q8_0 with max error {err}")

    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]}")

    out_file.close()
    print(f"wrote {filepath}")

# -----------------------------------------------------------------------------

def hf_export(model, params, filepath, group_size=64, dtype=jnp.float32):
    """ Generate the pytorch_model.bin state_dict and config.json for HuggingFace """
    
    try:
        import torch
        from transformers.models.llama.configuration_llama import LlamaConfig
    except ImportError:
        print("Error: transformers and torch packages are required to export to HuggingFace format")
        print("Please run `pip install transformers torch` to install them")
        return None

    flat_params = flatten_dict(params)
    model_config = model.config
    hf_state_dict = {}

    dim = model_config.dim
    num_key_value_heads = model_config.n_kv_heads if model_config.n_kv_heads is not None else model_config.n_heads
    n_rep = model_config.n_heads // num_key_value_heads
    key_value_dim = dim // n_rep

    def permute_original(w, n_heads=model_config.n_heads, dim1=dim, dim2=dim):
        w_torch = torch.tensor(np.array(w))
        return w_torch.view(dim1, dim2).reshape(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    tok_embeddings = flat_params.get(('tok_embeddings', 'embedding'))
    output_weights = flat_params.get(('output', 'kernel'))
    shared_classifier = jnp.array_equal(tok_embeddings, output_weights.T) if tok_embeddings is not None and output_weights is not None else False
    
    hf_state_dict['model.embed_tokens.weight'] = torch.tensor(np.array(tok_embeddings))
    hf_state_dict['model.norm.weight'] = torch.tensor(np.array(flat_params[('norm', 'scale')]))

    for i in range(model_config.n_layers):
        hf_state_dict[f'model.layers.{i}.input_layernorm.weight'] = torch.tensor(
            np.array(flat_params[('layers', f'{i}', 'attention_norm', 'scale')]))
        
        # Handle Q, K, V, O
        wq = flat_params[('layers', f'{i}', 'attention', 'wq', 'kernel')].T
        wk = flat_params[('layers', f'{i}', 'attention', 'wk', 'kernel')].T
        wv = flat_params[('layers', f'{i}', 'attention', 'wv', 'kernel')].T
        wo = flat_params[('layers', f'{i}', 'attention', 'wo', 'kernel')].T
        
        hf_state_dict[f'model.layers.{i}.self_attn.q_proj.weight'] = permute_original(wq)
        hf_state_dict[f'model.layers.{i}.self_attn.k_proj.weight'] = permute_original(wk, num_key_value_heads, key_value_dim, dim)
        hf_state_dict[f'model.layers.{i}.self_attn.v_proj.weight'] = torch.tensor(np.array(wv))
        hf_state_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = torch.tensor(np.array(wo))
        
        # Handle FFN
        hf_state_dict[f'model.layers.{i}.post_attention_layernorm.weight'] = torch.tensor(
            np.array(flat_params[('layers', f'{i}', 'ffn_norm', 'scale')]))
        
        w1 = flat_params[('layers', f'{i}', 'feed_forward', 'w1', 'kernel')].T
        w2 = flat_params[('layers', f'{i}', 'feed_forward', 'w2', 'kernel')].T
        w3 = flat_params[('layers', f'{i}', 'feed_forward', 'w3', 'kernel')].T
        
        hf_state_dict[f'model.layers.{i}.mlp.gate_proj.weight'] = torch.tensor(np.array(w1))
        hf_state_dict[f'model.layers.{i}.mlp.down_proj.weight'] = torch.tensor(np.array(w2))
        hf_state_dict[f'model.layers.{i}.mlp.up_proj.weight'] = torch.tensor(np.array(w3))

    hf_state_dict['lm_head.weight'] = hf_state_dict['model.embed_tokens.weight']

    if not shared_classifier:
        hf_state_dict['lm_head.weight'] = torch.tensor(np.array(output_weights.T))

    # Generate LlamaConfig
    vocab_size = model_config.vocab_size
    hidden_size = model_config.dim
    intermediate_size = flat_params[('layers', '0', 'feed_forward', 'w1', 'kernel')].shape[0]
    num_hidden_layers = model_config.n_layers
    num_attention_heads = model_config.n_heads
    num_key_value_heads = model_config.n_kv_heads if model_config.n_kv_heads is not None else model_config.n_heads
    max_position_embeddings = model_config.max_seq_len
    rms_norm_eps = model_config.norm_eps

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        tie_word_embeddings=shared_classifier,
        # Manual
        architectures=["LlamaForCausalLM"],
        hidden_act="silu",
    )
    os.makedirs(filepath, exist_ok=True)

    torch.save(hf_state_dict, os.path.join(filepath, "pytorch_model.bin"))
    config.save_pretrained(filepath)

# -----------------------------------------------------------------------------

def model_export(model, params, filepath, version, dtype=jnp.float32):
    """
    JAX version of model_export that exports model weights to various formats.
    
    Args:
        model: The JAX/Flax model instance
        params: The model parameters (typically from model.params)
        filepath: Output file path
        version: Export version to use
            v-1: huggingface export, i.e. intended for use outside of this repo, in HF
            v0: legacy llama2.c float format, DEPRECATED
            v1: float32 export
            v2: int8 quantized Q8_0 export, similar to llama.cpp, in groups
        dtype: Data type to use for export
    """
    if version == 0:
        legacy_export(model, params, filepath)
    elif version == 1:
        version1_export(model, params, filepath)
    elif version == 2:
        version2_export(model, params, filepath)
    elif version == -1:
        hf_export(model, params, filepath, dtype=dtype)
    else:
        raise ValueError(f"unknown version {version}")

def torchscript_export(model, params, filepath, zero_params=False, gzip_output=False):
    """
    This function is not directly applicable for JAX models as TorchScript is specific to PyTorch.
    For JAX models, consider using other serialization methods like pickle or saving to 
    a custom format using the provided export functions.
    """
    print("TorchScript export is not available for JAX models.")
    print("Please use one of the other export methods or save using native JAX serialization.")

# -----------------------------------------------------------------------------

def load_checkpoint_jax(checkpoint_path):
    """
    Load a checkpoint saved in JAX format
    """
    try:
        import pickle
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        return checkpoint_data['model'], checkpoint_data['params']
    except:
        print(f"Failed to load JAX checkpoint from {checkpoint_path}")
        return None, None

def convert_torch_to_jax(torch_checkpoint, model_class):
    """
    Convert a PyTorch checkpoint to JAX format
    This is a placeholder - actual implementation would depend on model structure
    """
    try:
        import torch
        checkpoint_dict = torch.load(torch_checkpoint, map_location='cpu')
        jax_config = {}
        if 'model_args' in checkpoint_dict:
            jax_config.update(checkpoint_dict['model_args'])
        jax_model = model_class(**jax_config)
        
        print("PyTorch to JAX conversion requires a custom implementation specific to your model architecture")
        return None, None
    except ImportError:
        print("PyTorch is required for this conversion.")
        return None, None

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument("--version", default=0, type=int, help="the version to export with")
    parser.add_argument("--dtype", type=str, help="dtype of the model (fp16, fp32)", default="fp32")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="JAX model checkpoint")
    args = parser.parse_args()
    
    dtype = {"fp16": jnp.float16, "fp32": jnp.float32}[args.dtype]
    if args.checkpoint:
        model, params = load_checkpoint_jax(args.checkpoint)
    else:
        parser.error("Must provide a checkpoint to load!")

    if model is None or params is None:
        parser.error("Can't load input model!")
    # Export the model
    model_export(model, params, args.filepath, args.version, dtype)