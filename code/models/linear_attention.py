import torch
import torch.nn as nn
import numpy as np

class SelfAttentionLayer(nn.Module):
    """
    Один слой Self-Attention с поддержкой Multi-Head и двух режимов работы.
    
    Linear Self-Attention (use_softmax=False):
        x <- x + W_o @ (Q @ K^T @ V)
        x <- x + MLP(x)  # если mlp_hidden_sizes не None
    
    Softmax Attention (use_softmax=True):
        x <- x + W_o @ softmax(Q @ K^T / sqrt(d_head)) @ V
        x <- x + MLP(x)  # если mlp_hidden_sizes не None
    
    Наследуется от nn.Module.
    
    Attributes:
        dim (int): Размерность токенов (nx+ny).
        use_softmax (bool): True - Softmax Attention, False - Linear Self-Attention.
        n_heads (int): Количество голов внимания.
        d_head (int): Размерность одной головы (dim // n_heads).
        mlp_hidden_sizes (list[int] | None): Размеры скрытых слоёв MLP.
        W_q, W_k, W_v (nn.Linear): Линейные проекции для Q, K, V без bias.
        W_o (nn.Linear): Выходная линейная проекция без bias.
        mlp (nn.Sequential | None): MLP блок.
    """

    def __init__(self, dim: int, use_softmax: bool = True, n_heads: int = 1, mlp_hidden_sizes: list[int] | None = None):
        """
        Инициализирует слой Self-Attention.
        
        Args:
            dim: Размерность токенов.
            use_softmax: True - Softmax Attention, False - Linear Self-Attention.
            n_heads: Количество голов. dim должно делиться на n_heads нацело.
        """

        super().__init__()

        assert dim % n_heads == 0, f'dim ({dim}) must be divisible by n_heads ({n_heads})'

        self.dim = dim
        self.use_softmax = use_softmax
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.mlp_hidden_sizes = mlp_hidden_sizes

        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.W_o = nn.Linear(self.dim, self.dim, bias=False)

        if mlp_hidden_sizes is not None:
            layers = []
            in_dim = dim
            for h_dim in mlp_hidden_sizes:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(nn.GELU())
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, dim))
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход слоя.
        
        Args:
            x: Токены формы (batch, seq_len, dim).
            
        Returns:
            torch.Tensor: Обновлённые токены формы (batch, seq_len, dim).
        """

        B, L, D = x.shape

        Q = self.W_q(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2) # (batch, n_heads, seq_len, d_head)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2) # (batch, n_heads, seq_len, d_head)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2) # (batch, n_heads, seq_len, d_head)

        attn_scores = Q @ K.transpose(-2, -1)  # (batch, n_heads, seq_len, d_head) @ (batch, n_heads, d_head, seq_len) --->
                                               # ---> (batch, n_heads, seq_len, seq_len)
        if self.use_softmax:
            attn_scores = attn_scores / (self.d_head ** 0.5)
            attn_weights = torch.softmax(attn_scores, dim=-1)
        else:
            attn_weights = attn_scores

        attn_out = attn_weights @ V # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, d_head) ---> 
                                    # ---> (batch, n_heads, seq_len, d_head)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D) # (batch, seq_len, dim)
        x = x + self.W_o(attn_out)

        if self.mlp is not None:
            x = x + self.mlp(x)

        return x

    def reset_parameters(self):
        """Переинициализирует все веса слоя."""
        self.W_q.reset_parameters()
        self.W_k.reset_parameters()
        self.W_v.reset_parameters()
        self.W_o.reset_parameters()
        if self.mlp is not None:
            for module in self.mlp:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()


class SelfAttentionModel(nn.Module):
    """
    Модель Self-Attention для экспериментов по In-Context Learning.
    
    Позволяет создавать модели разной глубины и конфигурации:
    - Однослойные/многослойные (n_layers)
    - Linear Self-Attention / Softmax Attention (use_softmax)
    - С LayerNorm между слоями или без (use_layernorm)
    
    Предсказание извлекается из y-части последнего токена
    с умножением на readout_scale.
    Наследуется от nn.Module.
    
    Attributes:
        nx (int): Размерность признаков x.
        ny (int): Размерность целевой переменной y.
        dim (int): Общая размерность токена (nx+ny).
        n_layers (int): Количество слоёв SelfAttentionLayer.
        use_softmax (bool): Тип внимания для всех слоёв.
        use_layernorm (bool): Применять LayerNorm между слоями.
        mlp_hidden_sizes (list[int] | None): Размеры скрытых слоёв MLP.
        readout_scale (float): Множитель при чтении предсказания.
        layers (nn.ModuleList): Список слоёв SelfAttentionLayer.
        norms (nn.ModuleList | None): Список LayerNorm (если use_layernorm=True).
    """

    def __init__(self,
        nx: int,
        ny: int,
        n_layers: int = 1,
        use_softmax: bool = True,
        use_layernorm: bool = True,
        mlp_hidden_sizes: list[int] | None = None,
        readout_scale: float = -1.0):

        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dim = nx + ny
        self.n_layers = n_layers
        self.use_softmax = use_softmax
        self.use_layernorm = use_layernorm
        self.readout_scale = readout_scale

        if mlp_hidden_sizes is None:
            mlp_hidden_sizes = [None] * n_layers

        self.layers = nn.ModuleList([
            SelfAttentionLayer(self.dim, use_softmax=use_softmax, mlp_hidden_sizes=mlp_hidden_sizes[i])
            for i in range(n_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(self.dim)
            for _ in range(n_layers)
        ]) if use_layernorm else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через все слои.
        
        Args:
            x: Токены формы (batch, seq_len, dim).
            
        Returns:
            torch.Tensor: Преобразованные токены формы (batch, seq_len, dim).
        """

        for i, layer in enumerate(self.layers):
            if self.norms is not None:
                x = self.norms[i](x)
            x = layer(x)
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Извлекает предсказание для тестового токена.
        
        Берёт y-часть последнего токена и умножает на readout_scale.
        
        Args:
            x: Токены формы (batch, seq_len, dim).
            
        Returns:
            torch.Tensor: Предсказания формы (batch, ny).
        """

        out = self.forward(x)
        return self.readout_scale * out[:, -1, -self.ny:]
    
    def reset_parameters(self):
        """
        Переинициализирует веса всех слоёв и нормализаций.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

