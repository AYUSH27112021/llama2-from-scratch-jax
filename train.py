import math
import os
import time
from datetime import datetime
import argparse
from functools import partial
from typing import Any, Dict, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
from jax.example_libraries import optimizers
import numpy as np
import optax

# -----------------------------------------------------------------------------
# Default configuration
# -----------------------------------------------------------------------------

# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 100
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume'

# wandb logging
wandb_log = False  # disabled by default
wandb_project = "jax-llamac"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# data
batch_size = 128
max_seq_len = 256
vocab_source = "llama2"  # llama2|custom; use Llama 2 vocab from Meta, or custom trained
vocab_size = 32000  # the Llama 2 tokenizer has 32K tokens

# model
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0

# optimizer
learning_rate = 5e-4  # max learning rate
max_iters = 100000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 1000  # how many steps to warm up for

# -----------------------------------------------------------------------------
# Parse command line arguments
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer language model with JAX")
    
    # I/O
    parser.add_argument('--out_dir', type=str, default=out_dir)
    parser.add_argument('--eval_interval', type=int, default=eval_interval)
    parser.add_argument('--log_interval', type=int, default=log_interval)
    parser.add_argument('--eval_iters', type=int, default=eval_iters)
    parser.add_argument('--eval_only', action='store_true', default=eval_only)
    parser.add_argument('--always_save_checkpoint', action='store_true', default=always_save_checkpoint)
    parser.add_argument('--init_from', type=str, default=init_from, choices=['scratch', 'resume'])
    
    # wandb
    parser.add_argument('--wandb_log', action='store_true', default=wandb_log)
    parser.add_argument('--wandb_project', type=str, default=wandb_project)
    parser.add_argument('--wandb_run_name', type=str, default=wandb_run_name)
    
    # data
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--max_seq_len', type=int, default=max_seq_len)
    parser.add_argument('--vocab_source', type=str, default=vocab_source, choices=['llama2', 'custom'])
    parser.add_argument('--vocab_size', type=int, default=vocab_size)
    
    # model
    parser.add_argument('--dim', type=int, default=dim)
    parser.add_argument('--n_layers', type=int, default=n_layers)
    parser.add_argument('--n_heads', type=int, default=n_heads)
    parser.add_argument('--n_kv_heads', type=int, default=n_kv_heads)
    parser.add_argument('--multiple_of', type=int, default=multiple_of)
    parser.add_argument('--dropout', type=float, default=dropout)
    
    # optimizer
    parser.add_argument('--learning_rate', type=float, default=learning_rate)
    parser.add_argument('--max_iters', type=int, default=max_iters)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--beta1', type=float, default=beta1)
    parser.add_argument('--beta2', type=float, default=beta2)
    parser.add_argument('--grad_clip', type=float, default=grad_clip)
    
    # lr scheduler
    parser.add_argument('--decay_lr', action='store_true', default=decay_lr)
    parser.add_argument('--warmup_iters', type=int, default=warmup_iters)
    
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------

def create_rotary_embedding(dim, max_seq_len):
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))
    t = jnp.arange(max_seq_len)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(emb), jnp.sin(emb)

def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_embedding(q, k, cos, sin, position_ids):
    # Get the corresponding cos and sin values for each position
    cos = cos[position_ids][:, :, None, :]  # [batch, seq, 1, dim]
    sin = sin[position_ids][:, :, None, :]  # [batch, seq, 1, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def create_attention_mask(input_ids, pad_token_id=0):
    mask = (input_ids != pad_token_id)
    return mask[:, None, None, :]  # [batch, 1, 1, seq_len]

def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v: [batch, seq, heads, head_dim]
    # Compute attention scores
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(q.shape[-1])

    if mask is not None:
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)

    weights = jax.nn.softmax(scores, axis=-1)

    return jnp.matmul(weights, v)

def layer_norm(x, weight, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) / jnp.sqrt(var + eps) + bias

def mlp(x, c_fc, c_proj, dropout_key, dropout_rate, training):
    h = jnp.matmul(x, c_fc['weight']) + c_fc['bias']
    h = jax.nn.gelu(h)
    h = jnp.matmul(h, c_proj['weight']) + c_proj['bias']
    
    if training and dropout_rate > 0:
        h = jax.random.dropout(dropout_key, dropout_rate, h)
    
    return h

def attention(q, k, v, mask, dropout_key, dropout_rate, training):
    attn_output = scaled_dot_product_attention(q, k, v, mask)
    
    if training and dropout_rate > 0:
        attn_output = jax.random.dropout(dropout_key, dropout_rate, attn_output)
    
    return attn_output

def transformer_block(x, params, position_ids, cos, sin, mask, dropout_key, dropout_rate, training):
    ln1_out = layer_norm(x, params['ln_1']['weight'], params['ln_1']['bias'])

    q = jnp.matmul(ln1_out, params['attn']['q_proj']['weight']) + params['attn']['q_proj']['bias']
    k = jnp.matmul(ln1_out, params['attn']['k_proj']['weight']) + params['attn']['k_proj']['bias']
    v = jnp.matmul(ln1_out, params['attn']['v_proj']['weight']) + params['attn']['v_proj']['bias']

    batch_size, seq_len, _ = q.shape
    head_dim = params['attn']['q_proj']['weight'].shape[1] // params['n_heads']
    
    q = q.reshape(batch_size, seq_len, params['n_heads'], head_dim)
    k = k.reshape(batch_size, seq_len, params['n_heads'], head_dim)
    v = v.reshape(batch_size, seq_len, params['n_heads'], head_dim)

    q, k = apply_rotary_embedding(q, k, cos, sin, position_ids)

    attn_dropout_key = random.split(dropout_key)[0]
    attn_output = attention(q, k, v, mask, attn_dropout_key, dropout_rate, training)

    attn_output = attn_output.reshape(batch_size, seq_len, -1)

    attn_output = jnp.matmul(attn_output, params['attn']['o_proj']['weight']) + params['attn']['o_proj']['bias']
    
    if training and dropout_rate > 0:
        proj_dropout_key = random.split(dropout_key)[1]
        attn_output = jax.random.dropout(proj_dropout_key, dropout_rate, attn_output)
    
    x = x + attn_output

    ln2_out = layer_norm(x, params['ln_2']['weight'], params['ln_2']['bias'])

    mlp_dropout_key = random.split(dropout_key)[2]
    mlp_output = mlp(ln2_out, params['mlp']['c_fc'], params['mlp']['c_proj'], 
                     mlp_dropout_key, dropout_rate, training)
    
    return x + mlp_output

def transformer_model(params, input_ids, position_ids, cos, sin, mask, dropout_key, dropout_rate, training=True):
    h = params['wte']['weight'][input_ids]
    
    dropout_keys = random.split(dropout_key, params['n_layers'] + 1)
    
    for i in range(params['n_layers']):
        layer_dropout_key = dropout_keys[i]
        h = transformer_block(h, params['h'][i], position_ids, cos, sin, mask, 
                             layer_dropout_key, dropout_rate, training)
    
    h = layer_norm(h, params['ln_f']['weight'], params['ln_f']['bias'])
    
    return jnp.matmul(h, params['lm_head']['weight'])

def create_position_ids(input_ids, pad_token_id=0):
    mask = (input_ids != pad_token_id).astype(jnp.int32)

    position_ids = jnp.cumsum(mask, axis=1) * mask - 1

    position_ids = jnp.maximum(position_ids, 0)
    
    return position_ids

def cross_entropy_loss(logits, targets, ignore_index=-100):
    vocab_size = logits.shape[-1]
    
    logits = logits[:, :-1, :]  # [batch, seq-1, vocab]
    targets = targets[:, 1:]    # [batch, seq-1]
    
    one_hot_targets = jax.nn.one_hot(targets, vocab_size)
    loss = -jnp.sum(one_hot_targets * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    mask = (targets != ignore_index).astype(logits.dtype)
    loss = jnp.sum(loss * mask) / jnp.sum(mask)
    
    return loss

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def init_params(config, rng_key):
    """Initialize model parameters"""
    head_dim = config['dim'] // config['n_heads']
    
    params = {
        'n_layers': config['n_layers'],
        'n_heads': config['n_heads'],
        'wte': {
            'weight': jax.random.normal(rng_key, (config['vocab_size'], config['dim'])) * 0.02,
        },
        'h': [],
        'ln_f': {
            'weight': jnp.ones(config['dim']),
            'bias': jnp.zeros(config['dim']),
        },
        'lm_head': {
            'weight': jnp.transpose(jax.random.normal(rng_key, (config['vocab_size'], config['dim'])) * 0.02),
        }
    }
    
    for i in range(config['n_layers']):
        layer_key = random.split(rng_key)[i]
        layer = {
            'ln_1': {
                'weight': jnp.ones(config['dim']),
                'bias': jnp.zeros(config['dim']),
            },
            'ln_2': {
                'weight': jnp.ones(config['dim']),
                'bias': jnp.zeros(config['dim']),
            },
            'attn': {
                'q_proj': {
                    'weight': jax.random.normal(layer_key, (config['dim'], config['dim'])) * 0.02,
                    'bias': jnp.zeros(config['dim']),
                },
                'k_proj': {
                    'weight': jax.random.normal(layer_key, (config['dim'], config['dim'])) * 0.02,
                    'bias': jnp.zeros(config['dim']),
                },
                'v_proj': {
                    'weight': jax.random.normal(layer_key, (config['dim'], config['dim'])) * 0.02,
                    'bias': jnp.zeros(config['dim']),
                },
                'o_proj': {
                    'weight': jax.random.normal(layer_key, (config['dim'], config['dim'])) * 0.02,
                    'bias': jnp.zeros(config['dim']),
                },
            },
            'mlp': {
                'c_fc': {
                    'weight': jax.random.normal(layer_key, (config['dim'], 4 * config['dim'])) * 0.02,
                    'bias': jnp.zeros(4 * config['dim']),
                },
                'c_proj': {
                    'weight': jax.random.normal(layer_key, (4 * config['dim'], config['dim'])) * 0.02,
                    'bias': jnp.zeros(config['dim']),
                },
            }
        }
        params['h'].append(layer)
    
    return params

def create_learning_rate_scheduler(config):
    """Create learning rate scheduler function"""
    def lr_schedule(step):
        if step < config['warmup_iters']:
            return config['learning_rate'] * step / config['warmup_iters']

        if config['decay_lr']:
            decay_ratio = (step - config['warmup_iters']) / (config['max_iters'] - config['warmup_iters'])
            decay_ratio = min(max(0.0, decay_ratio), 1.0)
            coeff = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio))
            return config['learning_rate'] * coeff

        return config['learning_rate']
    
    return lr_schedule

def create_optimizer(config, params):
    """Create optimizer with learning rate schedule"""
    lr_schedule = create_learning_rate_scheduler(config)

    optimizer = optax.chain(
        optax.clip_by_global_norm(config['grad_clip']),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=config['beta1'],
            b2=config['beta2'],
            weight_decay=config['weight_decay']
        )
    )
    
    opt_state = optimizer.init(params)
    return optimizer, opt_state

@jit
def train_step(params, opt_state, optimizer, rng_key, batch, cos, sin):
    """Single training step"""
    input_ids, targets = batch
    
    # position IDs and attention mask
    position_ids = create_position_ids(input_ids)
    attention_mask = create_attention_mask(input_ids)
    
    dropout_key = random.split(rng_key)[0]
    
    def loss_fn(params):
        logits = transformer_model(
            params, input_ids, position_ids, cos, sin, 
            attention_mask, dropout_key, config['dropout'], training=True
        )
        return cross_entropy_loss(logits, targets)
    
    # Compute loss and gradients
    loss, grads = value_and_grad(loss_fn)(params)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss

@jit
def eval_step(params, rng_key, batch, cos, sin):
    """Evaluation step"""
    input_ids, targets = batch
    
    position_ids = create_position_ids(input_ids)
    attention_mask = create_attention_mask(input_ids)
    
    logits = transformer_model(
        params, input_ids, position_ids, cos, sin, 
        attention_mask, rng_key, 0.0, training=False
    )
    
    loss = cross_entropy_loss(logits, targets)
    return loss

def estimate_loss(params, rng_key, data_iter_fn, config, cos, sin):
    """Estimate loss on train and val sets"""
    losses = {'train': 0.0, 'val': 0.0}
    
    for split in ['train', 'val']:
        total_loss = 0.0
        for _ in range(config['eval_iters']):
            rng_key, subkey = random.split(rng_key)
            batch = data_iter_fn(split)
            loss = eval_step(params, subkey, batch, cos, sin)
            total_loss += loss
        
        losses[split] = total_loss / config['eval_iters']
    
    return losses

def save_checkpoint(params, opt_state, config, iter_num, best_val_loss, out_dir):
    """Save model checkpoint"""
    import pickle
    
    checkpoint = {
        'params': params,
        'opt_state': opt_state,
        'config': config,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss
    }
    
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'checkpoint.pkl'), 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved to {out_dir}")

def load_checkpoint(checkpoint_path):
    """Load model checkpoint"""
    import pickle
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    return checkpoint

def main():
    args = parse_args()
    global config
    config = vars(args)
    config['lr_decay_iters'] = config['max_iters']
    
    rng_key = random.PRNGKey(1337)
    
    def mock_data_iter(split):
        """Generate random batches for testing"""
        rng = np.random.RandomState(42 if split == 'val' else 1337)
        while True:
            input_ids = rng.randint(0, config['vocab_size'], 
                                   (config['batch_size'], config['max_seq_len']))
            targets = np.roll(input_ids, -1, axis=1)
            targets[:, -1] = -100  # Mask last token
            yield jnp.array(input_ids), jnp.array(targets)
    
    def get_batch(split):
        return next(mock_data_iter(split))

    if config['init_from'] == 'scratch':
        print("Initializing model from scratch")
        rng_key, init_key = random.split(rng_key)
        params = init_params(config, init_key)
        iter_num = 0
        best_val_loss = float('inf')
    else:
        print(f"Loading model from {config['out_dir']}")
        checkpoint = load_checkpoint(os.path.join(config['out_dir'], 'checkpoint.pkl'))
        params = checkpoint['params']
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        for k in ['dim', 'n_layers', 'n_heads', 'n_kv_heads', 'vocab_size', 'multiple_of', 'max_seq_len']:
            if k in checkpoint['config']:
                config[k] = checkpoint['config'][k]
    
    optimizer, opt_state = create_optimizer(config, params)
    if config['init_from'] == 'resume' and 'opt_state' in checkpoint:
        opt_state = checkpoint['opt_state']
    
    cos, sin = create_rotary_embedding(config['dim'] // config['n_heads'], config['max_seq_len'])
    
    if config['wandb_log']:
        try:
            import wandb
            wandb.init(project=config['wandb_project'], name=config['wandb_run_name'], config=config)
        except ImportError:
            print("WandB not installed. Continuing without logging.")
            config['wandb_log'] = False
    
    tokens_per_iter = config['batch_size'] * config['max_seq_len']
    print(f"Tokens per iteration: {tokens_per_iter:,}")
    
    if config['eval_only']:
        losses = estimate_loss(params, rng_key, get_batch, config, cos, sin)
        print(f"Eval only mode: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        return
    
    t0 = time.time()
    train_iter = mock_data_iter('train')
    
    print(f"Beginning training for {config['max_iters']} iterations")
    
    while iter_num < config['max_iters']:
        if iter_num % config['log_interval'] == 0:
            lr = create_learning_rate_scheduler(config)(iter_num)
            print(f"Step {iter_num}: lr = {lr:.2e}")
        
        if iter_num % config['eval_interval'] == 0:
            losses = estimate_loss(params, rng_key, get_batch, config, cos, sin)
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if config['wandb_log']:
                wandb.log({
                    'iter': iter_num,
                    'train/loss': losses['train'],
                    'val/loss': losses['val'],
                    'lr': create_learning_rate_scheduler(config)(iter_num),
                })
            
            if losses['val'] < best_val_loss or config['always_save_checkpoint']:
                best_val_loss = min(best_val_loss, losses['val'])
                save_checkpoint(params, opt_state, config, iter_num, best_val_loss, config['out_dir'])
        
        batch = next(train_iter)

        rng_key, train_key = random.split(rng_key)
        params, opt_state, loss = train_step(params, opt_state, optimizer, train_key, batch, cos, sin)

        if iter_num % config['log_interval'] == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            
            print(f"Step {iter_num}: loss {loss:.4f}, {dt*1000:.2f}ms")
        
        iter_num += 1
    
    # Final evaluation
    losses = estimate_loss(params, rng_key, get_batch, config, cos, sin)
    print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # final checkpoint
    save_checkpoint(params, opt_state, config, iter_num, best_val_loss, config['out_dir'])
    
    print("Training complete!")

if __name__ == "__main__":
    main()

# """
# Example usage:
# $ python jax_train.py --eval_iters=10 --batch_size=8
# """
