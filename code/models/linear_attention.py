import torch
import torch.nn as nn
import numpy as np
from .base import BaseICLModel

class LinearSelfAttentionLayer(nn.Module):
    """
    Слой Linear Self-Attention без softmax-нормализации.
    
    Реализует обновление токенов по формуле:
        x <- x + P @ Q @ K^T @ V
    Наследуется от nn.Module
    
    Attributes:
        dim (int): Размерность токенов (nx+ny).
        W_q, W_k, W_v, P (nn.Linear): Линейные проекции без bias.
    """

    def __init__(self, dim):
        """
        Инициализирует слой Linear Self-Attention.
        
        Args:
            dim: Размерность входных и выходных токенов.
        """

        super().__init__()
        self.dim = dim
    
        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.P = nn.Linear(self.dim, self.dim, bias=False)
    
    def forward(self, x):
        """
        Прямой проход слоя.
        
        Args:
            x: Токены формы (batch, seq_len, dim).
            
        Returns:
            torch.Tensor: Обновлённые токены формы (batch, seq_len, dim).
        """

        Q = self.W_q(x) # [batch, seq_len, dim]
        K = self.W_k(x) # [batch, seq_len, dim]
        V = self.W_v(x) # [batch, seq_len, dim]

        attn_scores = Q @ K.transpose(-2, -1)  # [batch, seq_len, dim] @ [batch, dim, seq_len] ---> [batch, seq_len, seq_len]
        attn_out = attn_scores @ V # [batch, seq_len, seq_len] @ [batch, seq_len, dim] ---> [batch, seq_len, dim]
        out = self.P(attn_out) # [batch, seq_len, dim] @ [batch, dim, dim] ---> [batch, seq_len, dim]

        return x + out

    def set_weights(self, W_q, W_k, W_v, P):
        """
        Устанавливает веса слоя вручную.
        
        Args:
            W_q, W_k, W_v, P: Матрицы весов формы (dim, dim).
        """

        with torch.no_grad():
            self.W_q.weight.copy_(W_q)
            self.W_k.weight.copy_(W_k)
            self.W_v.weight.copy_(W_v)
            self.P.weight.copy_(P)


class SingleLayerLSA(BaseICLModel):
    """
    Однослойная Linear Self-Attention модель для In-Context Learning.
    
    Модель принимает токенизированную последовательность [x_i, y_i] для
    обучающих примеров и [x_test, -W0 @ x_test] для тестового запроса.
    После прямого прохода предсказание извлекается из y-части последнего
    токена с умножением на readout_scale = -1. Наследуется от BaseICLModel
    
    Attributes:
        nx (int): Размерность признаков x.
        ny (int): Размерность целевой переменной y.
        dim (int): Общая размерность токена (nx+ny).
        layer (LinearSelfAttentionLayer): Слой attention (линейный).
        readout_scale (float): Множитель при чтении предсказания.
    """

    def __init__(self, nx, ny):
        """
        Инициализирует однослойную LSA модель.
        
        Args:
            nx: Размерность признаков x.
            ny: Размерность целевой переменной y.
        """

        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dim = nx + ny

        self.layer = LinearSelfAttentionLayer(self.dim)

        self.readout_scale = -1.0

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через слой attention.
        
        Args:
            tokens: Токены формы (batch, seq_len, dim).
            
        Returns:
            torch.Tensor: Преобразованные токены формы (batch, seq_len, dim).
        """

        return self.layer(tokens)
    
    def predict(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Извлекает предсказание для тестового токена.
        
        Берёт y-часть последнего токена и умножает на readout_scale.
        
        Args:
            tokens: Токены формы (batch, seq_len, dim).
            
        Returns:
            torch.Tensor: Предсказания формы (batch, ny).
        """

        out = self.forward(tokens) # [batch, seq_len, dim]
        return self.readout_scale * out[:, -1, -self.ny:] # [batch, ny]
    
    def set_weights_from_gd(self, eta: float, N: int, W0: torch.Tensor = None):
        """
        Устанавливает веса модели в соответствии с одним шагом GD.
        
        Матрицы:
            W_q[:nx, :nx] = I
            W_k[:nx, :nx] = I
            W_v[nx:, :nx] = W0
            W_v[nx:, nx:] = -I
            P = (eta / N) * I
        
        Args:
            eta: Learning rate градиентного спуска.
            N: Количество обучающих примеров в контексте.
            W0: Начальные веса модели формы (ny, nx). Если None, используется 0.
        """
        
        if W0 is None:
            W0 = torch.zeros(self.ny, self.nx)

        d = self.dim
        nx = self.nx
        ny = self.ny
        
        W_q = torch.zeros(d, d)
        W_q[:nx, :nx] = torch.eye(nx)
        
        W_k = torch.zeros(d, d)
        W_k[:nx, :nx] = torch.eye(nx)
        
        W_v = torch.zeros(d, d)
        W_v[:nx, nx:] = -torch.eye(nx, ny)
        
        P = torch.zeros(d, d)
        P[nx:, :nx] = (eta / N) * torch.eye(ny, nx)
        
        self.layer.set_weights(W_q, W_k, W_v, P)


class SelfAttentionLayer(nn.Module):
    """
    Слой Self-Attention с softmax.
    
    Реализует обновление токенов по формуле:
        x <- x + P @ SoftMax(Q @ K^T) @ V
    Наследуется от nn.Module
    
    Attributes:
        dim (int): Размерность токенов (nx+ny).
        W_q, W_k, W_v, P (nn.Linear): Линейные проекции без bias.
    """

    def __init__(self, dim):
        """
        Инициализирует слой Self-Attention.
        
        Args:
            dim: Размерность входных и выходных токенов.
        """

        super().__init__()
        self.dim = dim
    
        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.P = nn.Linear(self.dim, self.dim, bias=False)
    
    def forward(self, x):
        """
        Прямой проход слоя.
        
        Args:
            x: Токены формы (batch, seq_len, dim).
            
        Returns:
            torch.Tensor: Обновлённые токены формы (batch, seq_len, dim).
        """

        Q = self.W_q(x) # [batch, seq_len, dim]
        K = self.W_k(x) # [batch, seq_len, dim]
        V = self.W_v(x) # [batch, seq_len, dim]

        attn_scores = nn.Softmax(Q @ K.transpose(-2, -1))  # [batch, seq_len, dim] @ [batch, dim, seq_len] ---> [batch, seq_len, seq_len]
        attn_out = attn_scores @ V # [batch, seq_len, seq_len] @ [batch, seq_len, dim] ---> [batch, seq_len, dim]
        out = self.P(attn_out) # [batch, seq_len, dim] @ [batch, dim, dim] ---> [batch, seq_len, dim]

        return x + out


class SingleLayerSelfAttention(BaseICLModel):
    """
    Однослойная Self-Attention модель для In-Context Learning.
    
    Модель принимает токенизированную последовательность [x_i, y_i] для
    обучающих примеров и [x_test, -W0 @ x_test] для тестового запроса.
    После прямого прохода предсказание извлекается из y-части последнего
    токена с умножением на readout_scale = -1. Наследуется от BaseICLModel
    
    Attributes:
        nx (int): Размерность признаков x.
        ny (int): Размерность целевой переменной y.
        dim (int): Общая размерность токена (nx+ny).
        layer (SelfAttentionLayer): Слой attention.
        readout_scale (float): Множитель при чтении предсказания.
    """

    def __init__(self, nx, ny):
        """
        Инициализирует однослойную SA модель.
        
        Args:
            nx: Размерность признаков x.
            ny: Размерность целевой переменной y.
        """

        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dim = nx + ny

        self.layer = SelfAttentionLayer(self.dim)

        self.readout_scale = -1.0

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через слой attention.
        
        Args:
            tokens: Токены формы (batch, seq_len, dim).
            
        Returns:
            torch.Tensor: Преобразованные токены формы (batch, seq_len, dim).
        """

        return self.layer(tokens)
    
    def predict(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Извлекает предсказание для тестового токена.
        
        Берёт y-часть последнего токена и умножает на readout_scale.
        
        Args:
            tokens: Токены формы (batch, seq_len, dim).
            
        Returns:
            torch.Tensor: Предсказания формы (batch, ny).
        """

        out = self.forward(tokens) # [batch, seq_len, dim]
        return self.readout_scale * out[:, -1, -self.ny:] # [batch, ny]