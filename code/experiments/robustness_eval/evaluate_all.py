import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from code.models.transformer import ICLTransformer
from code.data.linear import LinearTaskGenerator
from code.experiments.analytical_gd import mse_gd_step, huber_gd_step
from code.experiments.robustness_eval.eval_matching import apply_gd_hint

def train_model(model: nn.Module, generator: LinearTaskGenerator, steps: int = 1000, lr: float = 1e-4, max_norm: float = 0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
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

def extract_weights_ols(model: nn.Module, generator: LinearTaskGenerator, X_ctx: torch.Tensor, Y_ctx: torch.Tensor, num_probes: int = 200, batch_size: int = 64):
    device = X_ctx.device
    x_dim = generator.x_dim
    
    all_X_test = []
    all_preds = []
    
    for start_idx in range(0, num_probes, batch_size):
        end_idx = min(start_idx + batch_size, num_probes)
        curr_batch_size = end_idx - start_idx
        
        X_test = torch.randn(curr_batch_size, 1, x_dim, device=device)
        X_ctx_rep = X_ctx.repeat(curr_batch_size, 1, 1)
        Y_ctx_rep = Y_ctx.repeat(curr_batch_size, 1, 1)
        
        context_tokens = torch.cat([X_ctx_rep, Y_ctx_rep], dim=-1)
        zeros = torch.zeros(curr_batch_size, 1, 1, device=device)
        test_tokens = torch.cat([X_test, zeros], dim=-1)
        
        sequence = torch.cat([context_tokens, test_tokens], dim=1)
        
        with torch.no_grad():
            preds = model(sequence) # (curr_batch_size,)
            
        all_X_test.append(X_test.squeeze(1)) # (curr_batch_size, x_dim)
        all_preds.append(preds.unsqueeze(-1)) # (curr_batch_size, 1)
        
    A = torch.cat(all_X_test, dim=0) # (num_probes, x_dim)
    B = torch.cat(all_preds, dim=0)  # (num_probes, 1)
    
    # OLS reconstruction: A * theta = B
    theta_tf, _, _, _ = torch.linalg.lstsq(A, B)
    return theta_tf # (x_dim, 1)

def evaluate_probing_ols(model: nn.Module, generator: LinearTaskGenerator, eval_steps: int = 50):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    total_l2_mse, total_l2_huber = 0.0, 0.0
    total_cos_mse, total_cos_huber = 0.0, 0.0
    
    # Best parameters from line search
    lr = 1.90
    delta = 4.00
    
    with torch.no_grad():
        for _ in range(eval_steps):
            X_context, Y_context, _, _ = generator.generate()
            X_ctx = X_context[0:1].to(device)
            Y_ctx = Y_context[0:1].to(device)
            
            theta_tf = extract_weights_ols(model, generator, X_ctx, Y_ctx, num_probes=200, batch_size=64)
            
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
            
            norm_tf = torch.norm(theta_tf) + 1e-8
            cos_mse = max([(torch.dot(theta_tf.view(-1), t.view(-1)) / (norm_tf * (torch.norm(t) + 1e-8))).item() for t in mse_traj])
            cos_huber = max([(torch.dot(theta_tf.view(-1), t.view(-1)) / (norm_tf * (torch.norm(t) + 1e-8))).item() for t in huber_traj])
            
            total_l2_mse += l2_mse
            total_l2_huber += l2_huber
            total_cos_mse += cos_mse
            total_cos_huber += cos_huber
            
    return (total_l2_mse / eval_steps, total_l2_huber / eval_steps, 
            total_cos_mse / eval_steps, total_cos_huber / eval_steps)

def main():
    x_dim = 10
    d_model = x_dim + 1
    batch_size = 64
    seq_len = 20
    train_steps = 1000
    
    # 1. Initialize models
    print("Initializing models...")
    model_baseline_clean = ICLTransformer(d_model=d_model, n_layers=1, n_heads=1, use_softmax=False, use_mlp=False, use_layernorm=False)
    model_lsa = ICLTransformer(d_model=d_model, n_layers=1, n_heads=1, use_softmax=False, use_mlp=False, use_layernorm=False)
    
    model_softmax = ICLTransformer(d_model=d_model, n_layers=3, n_heads=1, use_softmax=True, use_mlp=False, use_layernorm=False)
    model_full = ICLTransformer(d_model=d_model, n_layers=3, n_heads=1, use_softmax=True, use_mlp=True, use_layernorm=False)
    
    # Apply hint to models trained on outliers
    apply_gd_hint(model_lsa)
    apply_gd_hint(model_softmax)
    apply_gd_hint(model_full)
    
    generator_outliers = LinearTaskGenerator(x_dim=x_dim, batch_size=batch_size, seq_len=seq_len, noise_std=0.1, outlier_prob=0.1, outlier_scale=10.0)
    generator_clean = LinearTaskGenerator(x_dim=x_dim, batch_size=batch_size, seq_len=seq_len, noise_std=0.1, outlier_prob=0.0, outlier_scale=0.0)
    
    # 2. Train
    print("Training Baseline Clean (LSA)...")
    train_model(model_baseline_clean, generator_clean, steps=train_steps)
    print("Training LSA...")
    train_model(model_lsa, generator_outliers, steps=train_steps)
    print("Training LSA+Softmax...")
    train_model(model_softmax, generator_outliers, steps=train_steps)
    print("Training Full Transformer...")
    train_model(model_full, generator_outliers, steps=train_steps)
    
    # 3. Evaluate Metrics
    models = {
        "Baseline Clean (LSA)": model_baseline_clean,
        "LSA": model_lsa,
        "LSA + Softmax": model_softmax,
        "Full Transformer": model_full
    }
    
    results = {}
    baseline_clean_mse = evaluate_model(model_baseline_clean, generator_clean, eval_steps=100)
    
    for name, model in models.items():
        print(f"Evaluating and Probing {name}...")
        clean_mse = evaluate_model(model, generator_clean, eval_steps=100)
        rob_tax = clean_mse - baseline_clean_mse
        l2_mse, l2_huber, cos_mse, cos_huber = evaluate_probing_ols(model, generator_outliers, eval_steps=50)
        
        results[name] = {
            "clean_mse": clean_mse,
            "rob_tax": rob_tax,
            "l2_mse": l2_mse,
            "l2_huber": l2_huber,
            "cos_mse": cos_mse,
            "cos_huber": cos_huber
        }
        
    # Format Table
    table_str = "Model Name                             | Clean-Target MSE | Robustness Tax | D_traj (MSE GD) | D_traj (Huber GD) | Max Cosine Sim (MSE) | Max Cosine Sim (Huber)\n"
    table_str += "-" * 155 + "\n"
    
    for name, metrics in results.items():
        table_str += f"{name:<38} | {metrics['clean_mse']:<16.4f} | {metrics['rob_tax']:<14.4f} | {metrics['l2_mse']:<15.4f} | {metrics['l2_huber']:<17.4f} | {metrics['cos_mse']:<20.4f} | {metrics['cos_huber']:<22.4f}\n"
        
    print("\n" + table_str)
    
    os.makedirs('results', exist_ok=True)
    with open('results/final_metrics.txt', 'w') as f:
        f.write(table_str)
    print("Final results saved to results/final_metrics.txt")

if __name__ == '__main__':
    main()
