import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import sys
import numpy as np

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
    
    for _ in tqdm(range(steps), desc="Training", leave=False):
        optimizer.zero_grad()
        X_context, Y_context, X_test, Y_test_true = generator.generate()
        X_context = X_context.to(device)
        Y_context = Y_context.to(device)
        X_test = X_test.to(device)
        Y_test_true = Y_test_true.to(device)
        
        sequence = generator._tokenize(X_context, Y_context, X_test).to(device)
        predictions = model(sequence)
        
        # Flatten both to 1D to prevent broadcasting bugs
        loss = nn.MSELoss()(predictions.view(-1), Y_test_true.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

def generate_val_set(generator: LinearTaskGenerator, eval_steps: int):
    """Generates a fixed validation set for fair comparisons across all models."""
    batches = []
    with torch.no_grad():
        for _ in range(eval_steps):
            X_ctx, Y_ctx, X_test, Y_test = generator.generate()
            batches.append((X_ctx, Y_ctx, X_test, Y_test))
    return batches

def evaluate_model_fixed(model: nn.Module, generator: LinearTaskGenerator, val_batches: list):
    """Evaluates the model strictly on the fixed validation batches."""
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    total_loss = 0.0
    total_r2 = 0.0
    with torch.no_grad():
        for X_ctx, Y_ctx, X_test, Y_test in val_batches:
            X_ctx = X_ctx.to(device)
            Y_ctx = Y_ctx.to(device)
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)
            
            sequence = generator._tokenize(X_ctx, Y_context=Y_ctx, X_test=X_test).to(device)
            predictions = model(sequence)
            
            y_true = Y_test.view(-1)
            y_pred = predictions.view(-1)
            loss = nn.MSELoss()(y_pred, y_true)
            total_loss += loss.item()
            
            # Mathematically rigorous R^2
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
            r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
            total_r2 += r2.item()
            
    return total_loss / len(val_batches), total_r2 / len(val_batches)

def extract_weights_ols(model: nn.Module, generator: LinearTaskGenerator, X_ctx: torch.Tensor, Y_ctx: torch.Tensor, num_probes: int = 200, batch_size: int = 64):
    """
    Note: Since Softmax and MLP introduce non-linearities with respect to the input, 
    OLS probing extracts the Best Linear Surrogate (First-order Taylor approximation) 
    of the Transformer's learned implicit algorithm.
    """
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
            preds = model(sequence)
            
        all_X_test.append(X_test.squeeze(1))
        all_preds.append(preds.unsqueeze(-1))
        
    A = torch.cat(all_X_test, dim=0)
    B = torch.cat(all_preds, dim=0)
    
    # OLS reconstruction: A * theta = B
    theta_tf, _, _, _ = torch.linalg.lstsq(A, B)
    return theta_tf

def calibrate_analytical_huber(model: nn.Module, generator: LinearTaskGenerator, val_batches: list, model_name: str, n_layers: int):
    print(f"\nCalibrating Golden Parameters for {model_name}...")
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    lr_range = np.arange(0.1, 3.1, 0.2)
    delta_range = np.arange(0.5, 20.5, 0.5)
    
    best_cos = -1.0
    best_lr = 1.0
    best_delta = 1.0
    
    # Subsample for speed
    calib_batches = val_batches[:20]
    contexts = []
    theta_tfs = []
    
    with torch.no_grad():
        for X_ctx, Y_ctx, _, _ in calib_batches:
            X_ctx_single = X_ctx[0:1].to(device)
            Y_ctx_single = Y_ctx[0:1].to(device)
            theta_tf = extract_weights_ols(model, generator, X_ctx_single, Y_ctx_single, num_probes=200, batch_size=64)
            contexts.append((X_ctx_single, Y_ctx_single))
            theta_tfs.append(theta_tf)
            
    for lr in lr_range:
        for delta in delta_range:
            total_cos = 0.0
            for i in range(len(calib_batches)):
                X_ctx_single, Y_ctx_single = contexts[i]
                theta_tf = theta_tfs[i]
                
                theta_huber = torch.zeros(1, generator.x_dim, 1, device=device)
                huber_traj = []
                for _ in range(n_layers):
                    theta_huber = huber_gd_step(X_ctx_single, Y_ctx_single, theta_huber, delta=float(delta), lr=float(lr))
                    huber_traj.append(theta_huber.squeeze(0))
                
                norm_tf = torch.norm(theta_tf) + 1e-8
                cos_huber = max([(torch.dot(theta_tf.view(-1), t.view(-1)) / (norm_tf * (torch.norm(t) + 1e-8))).item() for t in huber_traj])
                total_cos += cos_huber
                
            avg_cos = total_cos / len(calib_batches)
            if avg_cos > best_cos:
                best_cos = avg_cos
                best_lr = lr
                best_delta = delta
                
    print(f"Golden Parameters Found for {model_name}! LR: {best_lr:.2f}, Delta: {best_delta:.2f} (Max Cosine at k={n_layers}: {best_cos:.4f})")
    return float(best_lr), float(best_delta)

def analyze_trajectory_dynamics(model: nn.Module, generator: LinearTaskGenerator, val_batches: list, golden_lr: float, golden_delta: float, n_layers: int, model_name: str):
    """
    Analyzes Huber GD trajectory up to 30 steps to find 'Acceleration' factors.
    Finds k_max_cos and k_match_loss dynamically.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    max_steps = 30
    
    total_tf_loss = 0.0
    total_huber_losses = [0.0] * max_steps
    total_cos_traj = [0.0] * max_steps
    
    eval_steps_probing = min(20, len(val_batches))
    
    with torch.no_grad():
        # First calculate losses over the entire fixed validation batch
        for X_ctx, Y_ctx, X_test, Y_test in val_batches:
            X_ctx = X_ctx.to(device)
            Y_ctx = Y_ctx.to(device)
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)
            
            # TF Loss
            sequence = generator._tokenize(X_ctx, Y_ctx, X_test).to(device)
            predictions = model(sequence)
            y_true = Y_test.view(-1)
            y_pred = predictions.view(-1)
            tf_loss = nn.MSELoss()(y_pred, y_true).item()
            total_tf_loss += tf_loss
            
            # Huber Loss Trajectory
            theta_huber = torch.zeros(generator.batch_size, generator.x_dim, 1, device=device)
            for k in range(max_steps):
                theta_huber = huber_gd_step(X_ctx, Y_ctx, theta_huber, delta=golden_delta, lr=golden_lr)
                huber_preds = torch.bmm(X_test, theta_huber).view(-1)
                h_loss = nn.MSELoss()(huber_preds, y_true).item()
                total_huber_losses[k] += h_loss
                
        tf_loss_avg = total_tf_loss / len(val_batches)
        huber_losses_avg = [l / len(val_batches) for l in total_huber_losses]
        
        # OLS Extract and track Cosine over trajectory
        for i in range(eval_steps_probing):
            X_ctx, Y_ctx, _, _ = val_batches[i]
            X_ctx_single = X_ctx[0:1].to(device)
            Y_ctx_single = Y_ctx[0:1].to(device)
            
            theta_tf = extract_weights_ols(model, generator, X_ctx_single, Y_ctx_single, num_probes=200, batch_size=64)
            norm_tf = torch.norm(theta_tf) + 1e-8
            
            theta_huber_single = torch.zeros(1, generator.x_dim, 1, device=device)
            for k in range(max_steps):
                theta_huber_single = huber_gd_step(X_ctx_single, Y_ctx_single, theta_huber_single, delta=golden_delta, lr=golden_lr)
                t = theta_huber_single.squeeze(0)
                cos = (torch.dot(theta_tf.view(-1), t.view(-1)) / (norm_tf * (torch.norm(t) + 1e-8))).item()
                total_cos_traj[k] += cos
                
        cos_traj_avg = [c / eval_steps_probing for c in total_cos_traj]
        
        # Calculate insight metrics
        huber_loss_at_n_layers = huber_losses_avg[n_layers - 1]
        
        k_max_cos = int(np.argmax(cos_traj_avg) + 1)
        max_cos = float(cos_traj_avg[k_max_cos - 1])
        
        # Closest matching loss
        loss_diffs = [abs(hl - tf_loss_avg) for hl in huber_losses_avg]
        k_match_loss = int(np.argmin(loss_diffs) + 1)
        
        accel_factor = k_match_loss / n_layers
        print(f"Insight for {model_name}: Loss matched after {k_match_loss} steps of analytical Huber GD, showing a {accel_factor:.1f}x acceleration factor.")
        
        return {
            "tf_loss": tf_loss_avg,
            "huber_loss_n_layers": huber_loss_at_n_layers,
            "max_cos": max_cos,
            "k_max_cos": k_max_cos,
            "k_match_loss": k_match_loss,
            "accel_factor": accel_factor
        }

def main():
    x_dim = 10
    d_model = x_dim + 1
    batch_size = 64
    seq_len = 20
    train_steps = 20000
    eval_steps = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mode = input("Do you want to (T)rain new models from scratch or (L)oad existing weights? [T/L]: ").strip().upper()
    
    # 1. Initialize paired models
    print("Initializing paired models...")
    
    model_lsa_clean = ICLTransformer(d_model=d_model, n_layers=1, n_heads=1, use_softmax=False, use_mlp=False, use_layernorm=False)
    model_lsa_outlier = ICLTransformer(d_model=d_model, n_layers=1, n_heads=1, use_softmax=False, use_mlp=False, use_layernorm=False)
    
    model_softmax_clean = ICLTransformer(d_model=d_model, n_layers=3, n_heads=1, use_softmax=True, use_mlp=False, use_layernorm=False)
    model_softmax_outlier = ICLTransformer(d_model=d_model, n_layers=3, n_heads=1, use_softmax=True, use_mlp=False, use_layernorm=False)
    
    model_full_clean = ICLTransformer(d_model=d_model, n_layers=3, n_heads=1, use_softmax=True, use_mlp=True, use_layernorm=False)
    model_full_outlier = ICLTransformer(d_model=d_model, n_layers=3, n_heads=1, use_softmax=True, use_mlp=True, use_layernorm=False)
    
    apply_gd_hint(model_lsa_clean)
    apply_gd_hint(model_lsa_outlier)
    apply_gd_hint(model_softmax_clean)
    apply_gd_hint(model_softmax_outlier)
    apply_gd_hint(model_full_clean)
    apply_gd_hint(model_full_outlier)
    
    models_dict = {
        'model_lsa_clean': model_lsa_clean,
        'model_lsa_outlier': model_lsa_outlier,
        'model_softmax_clean': model_softmax_clean,
        'model_softmax_outlier': model_softmax_outlier,
        'model_full_clean': model_full_clean,
        'model_full_outlier': model_full_outlier
    }
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    
    fallback_to_train = False
    if mode == 'L':
        print("Checking for existing checkpoints...")
        for name in models_dict.keys():
            ckpt_path = os.path.join(ckpt_dir, f"{name}.pth")
            if not os.path.exists(ckpt_path):
                print(f"Checkpoint not found: {ckpt_path}")
                fallback_to_train = True
                break
        
        if fallback_to_train:
            print("Falling back to training mode...")
            mode = 'T'
        else:
            print("Loading weights...")
            for name, model in models_dict.items():
                ckpt_path = os.path.join(ckpt_dir, f"{name}.pth")
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.eval()
    
    # Fixed Evaluation sets for Fair Comparison
    print("Generating fixed validation sets...")
    generator_outliers = LinearTaskGenerator(x_dim=x_dim, batch_size=batch_size, seq_len=seq_len, noise_std=0.1, outlier_prob=0.1, outlier_scale=10.0)
    generator_clean = LinearTaskGenerator(x_dim=x_dim, batch_size=batch_size, seq_len=seq_len, noise_std=0.1, outlier_prob=0.0, outlier_scale=0.0)
    
    val_batches_clean = generate_val_set(generator_clean, eval_steps)
    val_batches_outliers = generate_val_set(generator_outliers, eval_steps)
    
    if mode != 'L':
        # 2. Train pairs
        print(f"\nTraining Clean Models for {train_steps} steps...")
        train_model(model_lsa_clean, generator_clean, steps=train_steps)
        train_model(model_softmax_clean, generator_clean, steps=train_steps)
        train_model(model_full_clean, generator_clean, steps=train_steps)
        
        print(f"\nTraining Outlier Models for {train_steps} steps...")
        train_model(model_lsa_outlier, generator_outliers, steps=train_steps)
        train_model(model_softmax_outlier, generator_outliers, steps=train_steps)
        train_model(model_full_outlier, generator_outliers, steps=train_steps)
        
        print("Saving checkpoints...")
        os.makedirs(ckpt_dir, exist_ok=True)
        for name, model in models_dict.items():
            ckpt_path = os.path.join(ckpt_dir, f"{name}.pth")
            torch.save(model.state_dict(), ckpt_path)
    
    architectures = [
        ("LSA", model_lsa_clean, model_lsa_outlier, 1),
        ("LSA+Softmax", model_softmax_clean, model_softmax_outlier, 3),
        ("Full Transformer", model_full_clean, model_full_outlier, 3)
    ]
    
    results = {}
    
    for name, clean_model, outlier_model, n_layers in architectures:
        print(f"\n--- Analyzing {name} ---")
        
        # Base Clean MSE on Clean model
        clean_mse_from_clean, _ = evaluate_model_fixed(clean_model, generator_clean, val_batches_clean)
        
        # Target Clean MSE on Outlier model
        clean_mse_from_outlier, _ = evaluate_model_fixed(outlier_model, generator_clean, val_batches_clean)
        
        # Outlier MSE on Outlier model
        outlier_mse, _ = evaluate_model_fixed(outlier_model, generator_outliers, val_batches_outliers)
        
        # Architecture-specific Robustness Tax
        rob_tax = clean_mse_from_outlier - clean_mse_from_clean
        
        # Calibrate individual Golden Parameters
        lr, delta = calibrate_analytical_huber(outlier_model, generator_outliers, val_batches_outliers, name, n_layers)
        
        # Extended Step Search Analysis (Up to k=30)
        dyn = analyze_trajectory_dynamics(outlier_model, generator_outliers, val_batches_outliers, lr, delta, n_layers, name)
        
        results[name] = {
            "clean_mse": clean_mse_from_outlier,
            "outlier_mse": outlier_mse,
            "rob_tax": rob_tax,
            "lr": lr,
            "delta": delta,
            "max_cos": dyn["max_cos"],
            "k_max_cos": dyn["k_max_cos"],
            "k_match_loss": dyn["k_match_loss"],
            "tf_loss": dyn["tf_loss"],
            "huber_loss_n_layers": dyn["huber_loss_n_layers"]
        }
        
    table_str = "\nArchitecture         | Clean MSE | Outlier MSE | Robust Tax | Best Huber Params | Max Cosine | k_max_cos | k_match_loss | TF Loss | Huber Loss (at k=n_layers)\n"
    table_str += "-" * 165 + "\n"
    
    for name, metrics in results.items():
        params = f"LR={metrics['lr']:.1f}, d={metrics['delta']:.1f}"
        table_str += f"{name:<20} | {metrics['clean_mse']:<9.4f} | {metrics['outlier_mse']:<11.4f} | {metrics['rob_tax']:<10.4f} | {params:<17} | {metrics['max_cos']:<10.4f} | {metrics['k_max_cos']:<9} | {metrics['k_match_loss']:<12} | {metrics['tf_loss']:<7.4f} | {metrics['huber_loss_n_layers']:<26.4f}\n"
        
    print(table_str)
    
    os.makedirs('results', exist_ok=True)
    with open('results/comprehensive_final_metrics.txt', 'w') as f:
        f.write(table_str)
    print("Final comprehensive metrics saved to results/comprehensive_final_metrics.txt")

if __name__ == '__main__':
    main()
