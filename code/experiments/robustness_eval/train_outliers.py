import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from code.models.transformer import ICLTransformer
from code.data.linear import LinearTaskGenerator
from code.experiments.analytical_gd import mse_gd_step, huber_gd_step

def train_model(model: nn.Module, generator: LinearTaskGenerator, steps: int = 1000, lr: float = 1e-4, max_norm: float = 0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    losses = []
    for _ in tqdm(range(steps), desc="Training"):
        optimizer.zero_grad()
        
        X_context, Y_context, X_test, Y_test_true = generator.generate()
        
        X_context = X_context.to(device)
        Y_context = Y_context.to(device)
        X_test = X_test.to(device)
        Y_test_true = Y_test_true.squeeze(-1).to(device) 
        
        sequence = generator._tokenize(X_context, Y_context, X_test).to(device)
        predictions = model(sequence) 
        
        loss = nn.MSELoss()(predictions, Y_test_true.squeeze(-1))
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        losses.append(loss.item())
        
    return losses

def evaluate_model(model: nn.Module, generator: LinearTaskGenerator, eval_steps: int = 100):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_steps):
            X_context, Y_context, X_test, Y_test_true = generator.generate()
            
            X_context = X_context.to(device)
            Y_context = Y_context.to(device)
            X_test = X_test.to(device)
            Y_test_true = Y_test_true.squeeze(-1).to(device)
            
            sequence = generator._tokenize(X_context, Y_context, X_test).to(device)
            predictions = model(sequence)
            
            loss = nn.MSELoss()(predictions, Y_test_true.squeeze(-1))
            total_loss += loss.item()
            
    return total_loss / eval_steps

def extract_effective_weights(model: nn.Module, X_ctx: torch.Tensor, Y_ctx: torch.Tensor, generator: LinearTaskGenerator):
    """
    Extracts the context-conditioned effective weights by feeding the Identity matrix.
    X_ctx: (1, seq_len, x_dim)
    Y_ctx: (1, seq_len, 1)
    """
    device = X_ctx.device
    x_dim = X_ctx.shape[-1]
    
    # Repeat the context x_dim times to process all basis vectors at once
    X_ctx_rep = X_ctx.repeat(x_dim, 1, 1) # (x_dim, seq_len, x_dim)
    Y_ctx_rep = Y_ctx.repeat(x_dim, 1, 1) # (x_dim, seq_len, 1)
    
    # X_test is the identity matrix
    X_test = torch.eye(x_dim, device=device).unsqueeze(1) # (x_dim, 1, x_dim)
    
    # Tokenize
    context_tokens = torch.cat([X_ctx_rep, Y_ctx_rep], dim=-1) # (x_dim, seq_len, x_dim + 1)
    zeros = torch.zeros(x_dim, 1, 1, device=device)
    test_tokens = torch.cat([X_test, zeros], dim=-1) # (x_dim, 1, x_dim + 1)
    
    sequence = torch.cat([context_tokens, test_tokens], dim=1)
    
    # Pass through model
    predictions = model(sequence) # (x_dim,)
    
    # The predictions vector is exactly theta_tf
    theta_tf = predictions.view(x_dim, 1) # (x_dim, 1)
    return theta_tf

def evaluate_probing(model: nn.Module, generator: LinearTaskGenerator, eval_steps: int = 50):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    total_l2_mse, total_l2_huber = 0.0, 0.0
    total_cos_mse, total_cos_huber = 0.0, 0.0
    
    lr = 0.5 # Example learning rate for analytical baselines
    delta = 1.0 # Example delta for Huber loss
    
    with torch.no_grad():
        for _ in range(eval_steps):
            X_context, Y_context, _, _ = generator.generate()
            X_ctx = X_context[0:1].to(device)
            Y_ctx = Y_context[0:1].to(device)
            
            theta_tf = extract_effective_weights(model, X_ctx, Y_ctx, generator)
            
            theta_mse = torch.zeros(1, generator.x_dim, 1, device=device)
            theta_huber = torch.zeros(1, generator.x_dim, 1, device=device)
            
            mse_traj, huber_traj = [], []
            for _ in range(3):
                theta_mse = mse_gd_step(X_ctx, Y_ctx, theta_mse, lr=lr)
                theta_huber = huber_gd_step(X_ctx, Y_ctx, theta_huber, delta=delta, lr=lr)
                mse_traj.append(theta_mse.squeeze(0))
                huber_traj.append(theta_huber.squeeze(0))
                
            l2_mse = min([torch.norm(theta_tf - t).item() for t in mse_traj])
            l2_huber = min([torch.norm(theta_tf - t).item() for t in huber_traj])
            
            # Use small eps to avoid division by zero
            cos_mse = max([F.cosine_similarity(theta_tf.view(1, -1), t.view(1, -1)).item() for t in mse_traj])
            cos_huber = max([F.cosine_similarity(theta_tf.view(1, -1), t.view(1, -1)).item() for t in huber_traj])
            
            total_l2_mse += l2_mse
            total_l2_huber += l2_huber
            total_cos_mse += cos_mse
            total_cos_huber += cos_huber
            
    return (total_l2_mse / eval_steps, total_l2_huber / eval_steps, 
            total_cos_mse / eval_steps, total_cos_huber / eval_steps)

def main():
    # Architecture params
    x_dim = 10
    d_model = x_dim + 1
    batch_size = 64
    seq_len = 20
    
    train_steps = 1000 # Using 1000 steps to speed up execution, can be 2000
    eval_steps = 100
    probing_eval_steps = 50
    
    # 1. Initialize models (use_layernorm=False to ensure exact linear equivalence where needed)
    print("Initializing models...")
    model_baseline_clean = ICLTransformer(d_model=d_model, n_layers=1, n_heads=1, use_softmax=False, use_mlp=False, use_layernorm=False)
    model_lsa = ICLTransformer(d_model=d_model, n_layers=1, n_heads=1, use_softmax=False, use_mlp=False, use_layernorm=False)
    
    # 3 layers for softmax and full models to simulate iterative optimization
    model_softmax = ICLTransformer(d_model=d_model, n_layers=3, n_heads=1, use_softmax=True, use_mlp=False, use_layernorm=False)
    model_full = ICLTransformer(d_model=d_model, n_layers=3, n_heads=1, use_softmax=True, use_mlp=True, use_layernorm=False)
    
    # 2. Generators
    outlier_prob = 0.1
    outlier_scale = 10.0
    
    generator_outliers = LinearTaskGenerator(
        x_dim=x_dim, batch_size=batch_size, seq_len=seq_len, 
        noise_std=0.1, outlier_prob=outlier_prob, outlier_scale=outlier_scale
    )
    
    generator_clean = LinearTaskGenerator(
        x_dim=x_dim, batch_size=batch_size, seq_len=seq_len, 
        noise_std=0.1, outlier_prob=0.0, outlier_scale=0.0
    )
    
    # 3. Train models
    print("Training Baseline Clean LSA model on CLEAN data...")
    train_model(model_baseline_clean, generator_clean, steps=train_steps)
    
    print("\nTraining LSA model on OUTLIER data...")
    train_model(model_lsa, generator_outliers, steps=train_steps)
    
    print("\nTraining LSA + Softmax model on OUTLIER data...")
    train_model(model_softmax, generator_outliers, steps=train_steps)
    
    print("\nTraining Full Transformer model on OUTLIER data...")
    train_model(model_full, generator_outliers, steps=train_steps)
    
    # 4. Evaluation
    print("\n=== EVALUATION ===")
    
    models = {
        "Baseline Clean (LSA)": model_baseline_clean,
        "LSA (Trained w/ Outliers)": model_lsa,
        "LSA + Softmax (Trained w/ Outliers)": model_softmax,
        "Full Transformer (Trained w/ Outliers)": model_full
    }
    
    results_outliers = {}
    results_clean = {}
    results_probing = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        results_outliers[name] = evaluate_model(model, generator_outliers, eval_steps=eval_steps)
        results_clean[name] = evaluate_model(model, generator_clean, eval_steps=eval_steps)
        
        # We perform probing on the outlier distribution context
        results_probing[name] = evaluate_probing(model, generator_outliers, eval_steps=probing_eval_steps)
        
    print("\n--- PERFORMANCE RESULTS ---")
    print(f"{'Model':<40} | {'MSE (Outlier Data)':<20} | {'MSE (Clean Data)':<20} | {'Robustness Tax'}")
    print("-" * 105)
    
    baseline_clean_perf = results_clean["Baseline Clean (LSA)"]
    
    for name in models.keys():
        mse_out = results_outliers[name]
        mse_cln = results_clean[name]
        tax = mse_cln - baseline_clean_perf
        print(f"{name:<40} | {mse_out:<20.4f} | {mse_cln:<20.4f} | {tax:+.4f}")

    print("\n--- PROBING ALIGNMENT METRICS ---")
    print(f"{'Model':<40} | {'L2 to MSE':<12} | {'L2 to Huber':<12} | {'Cos to MSE':<12} | {'Cos to Huber'}")
    print("-" * 105)
    
    for name in models.keys():
        l2_mse, l2_huber, cos_mse, cos_huber = results_probing[name]
        print(f"{name:<40} | {l2_mse:<12.4f} | {l2_huber:<12.4f} | {cos_mse:<12.4f} | {cos_huber:+.4f}")

if __name__ == "__main__":
    main()
