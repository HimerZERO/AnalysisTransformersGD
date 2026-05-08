import torch
import torch.nn as nn
import math

class BaseICLModel(nn.Module):
    # Dummy base class to satisfy inheritance if not provided
    pass

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_softmax: bool = True, use_mlp: bool = True, use_layernorm: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_softmax = use_softmax
        self.use_mlp = use_mlp
        self.use_layernorm = use_layernorm
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.norm1 = nn.LayerNorm(d_model)
        
        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # Pre-LN
        x_norm = self.norm1(x) if self.use_layernorm else x
        
        # Multi-Head Attention
        q = self.W_q(x_norm).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x_norm).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x_norm).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # (B, n_heads, L, L)
        
        # Causal mask
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
        
        if self.use_softmax:
            attn_scores = attn_scores / math.sqrt(self.d_head)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=-1)
        else:
            # Purely linear (LSA), still apply causal mask but without softmax
            attn_weights = attn_scores.masked_fill(mask == 0, 0.0)

        attn_out = torch.matmul(attn_weights, v) # (B, n_heads, L, d_head)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.W_o(attn_out)
        
        x = x + attn_out
        
        # MLP
        if self.use_mlp:
            x_mlp_in = self.norm2(x) if self.use_layernorm else x
            x = x + self.mlp(x_mlp_in)
            
        return x

class ICLTransformer(BaseICLModel):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, use_softmax: bool = True, use_mlp: bool = True, use_layernorm: bool = False):
        super().__init__()
        self.d_model = d_model
        self.use_layernorm = use_layernorm
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, use_softmax, use_mlp, use_layernorm)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Token embeddings of shape (batch_size, seq_len, d_model)
        Returns: Prediction extracted from the y-portion of the final token
        """
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x) if self.use_layernorm else x
        
        # Prediction extraction: extracted from the y-portion of the final sequence token, multiplied by -1.0
        # In this implementation, we assume the y-portion is the last dimension of the embedding.
        final_token = x[:, -1, :] # (B, d_model)
        prediction = final_token[:, -1] * -1.0
        
        return prediction
