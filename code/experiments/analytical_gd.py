import torch

def mse_gd_step(X: torch.Tensor, Y: torch.Tensor, theta: torch.Tensor, lr: float) -> torch.Tensor:
    """
    Computes one step of Gradient Descent using MSE loss.
    X: (B, seq_len, x_dim)
    Y: (B, seq_len, 1)
    theta: (B, x_dim, 1)
    """
    # residuals: (B, seq_len, 1)
    residuals = torch.bmm(X, theta) - Y
    
    # Optional scaling: divide by sequence length
    seq_len = X.shape[1]
    
    # grad: (B, x_dim, 1)
    grad = torch.bmm(X.transpose(1, 2), residuals) / seq_len
    
    return theta - lr * grad

def huber_gd_step(X: torch.Tensor, Y: torch.Tensor, theta: torch.Tensor, delta: float, lr: float) -> torch.Tensor:
    """
    Computes one step of Gradient Descent using Huber loss.
    X: (B, seq_len, x_dim)
    Y: (B, seq_len, 1)
    theta: (B, x_dim, 1)
    """
    residuals = torch.bmm(X, theta) - Y
    clipped_residuals = torch.clamp(residuals, min=-delta, max=delta)
    
    seq_len = X.shape[1]
    
    # grad: (B, x_dim, 1)
    grad = torch.bmm(X.transpose(1, 2), clipped_residuals) / seq_len
    
    return theta - lr * grad
