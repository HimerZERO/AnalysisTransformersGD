import torch
import torch.nn.functional as F
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from code.models.transformer import ICLTransformer
from code.data.linear import LinearTaskGenerator
from code.experiments.analytical_gd import huber_gd_step
from code.experiments.robustness_eval.train_outliers import train_model, extract_effective_weights

def line_search_probing(model, generator, eval_steps=50):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    lr_range = np.arange(0.1, 3.1, 0.1)
    delta_range = np.arange(0.5, 5.5, 0.5)
    
    best_cos_huber = -1.0
    best_lr = None
    best_delta = None
    
    # Pre-extract weights for all eval_steps
    X_ctx_list = []
    Y_ctx_list = []
    theta_tf_list = []
    
    with torch.no_grad():
        for _ in range(eval_steps):
            X_context, Y_context, _, _ = generator.generate()
            X_ctx = X_context[0:1].to(device)
            Y_ctx = Y_context[0:1].to(device)
            
            theta_tf = extract_effective_weights(model, X_ctx, Y_ctx, generator)
            
            X_ctx_list.append(X_ctx)
            Y_ctx_list.append(Y_ctx)
            theta_tf_list.append(theta_tf)
            
    # Grid search
    print("Starting Grid Search over lr and delta...")
    for lr in lr_range:
        for delta in delta_range:
            total_cos = 0.0
            
            for i in range(eval_steps):
                X_ctx = X_ctx_list[i]
                Y_ctx = Y_ctx_list[i]
                theta_tf = theta_tf_list[i]
                
                theta_huber = torch.zeros(1, generator.x_dim, 1, device=device)
                
                huber_traj = []
                for _ in range(3):
                    theta_huber = huber_gd_step(X_ctx, Y_ctx, theta_huber, delta=float(delta), lr=float(lr))
                    huber_traj.append(theta_huber.squeeze(0))
                
                # Max cosine across the 3 steps
                # use small epsilon to avoid NaN on zero norm
                norm_tf = torch.norm(theta_tf) + 1e-8
                
                max_cos = -1.0
                for t in huber_traj:
                    norm_t = torch.norm(t) + 1e-8
                    cos = (torch.dot(theta_tf.view(-1), t.view(-1)) / (norm_tf * norm_t)).item()
                    if cos > max_cos:
                        max_cos = cos
                
                total_cos += max_cos
                
            avg_cos = total_cos / eval_steps
            if avg_cos > best_cos_huber:
                best_cos_huber = avg_cos
                best_lr = lr
                best_delta = delta
                
    return best_lr, best_delta, best_cos_huber

def apply_gd_hint(model):
    layer = model.layers[0]
    d_model = layer.d_model
    x_dim = d_model - 1
    
    with torch.no_grad():
        W_q = torch.zeros(d_model, d_model)
        W_q[:x_dim, :x_dim] = torch.eye(x_dim)
        layer.W_q.weight.copy_(W_q + torch.randn_like(W_q) * 0.01)
        
        W_k = torch.zeros(d_model, d_model)
        W_k[:x_dim, :x_dim] = torch.eye(x_dim)
        layer.W_k.weight.copy_(W_k + torch.randn_like(W_k) * 0.01)
        
        W_v = torch.zeros(d_model, d_model)
        W_v[-1, -1] = 1.0
        layer.W_v.weight.copy_(W_v + torch.randn_like(W_v) * 0.01)
        
        W_o = torch.eye(d_model)
        layer.W_o.weight.copy_(W_o + torch.randn_like(W_o) * 0.01)

def main():
    x_dim = 10
    d_model = x_dim + 1
    batch_size = 64
    seq_len = 20
    train_steps = 1000
    
    print("Training Full Transformer...")
    model_full = ICLTransformer(d_model=d_model, n_layers=3, n_heads=1, use_softmax=True, use_mlp=True, use_layernorm=False)
    apply_gd_hint(model_full)
    
    generator_outliers = LinearTaskGenerator(
        x_dim=x_dim, batch_size=batch_size, seq_len=seq_len, 
        noise_std=0.1, outlier_prob=0.1, outlier_scale=10.0
    )
    
    train_model(model_full, generator_outliers, steps=train_steps)
    
    print("Running Line Search...")
    best_lr, best_delta, best_cos = line_search_probing(model_full, generator_outliers, eval_steps=50)
    
    print(f"Optimal LR: {best_lr:.2f}")
    print(f"Optimal Delta: {best_delta:.2f}")
    print(f"Maximized Cosine Similarity: {best_cos:.4f}")

if __name__ == '__main__':
    main()
