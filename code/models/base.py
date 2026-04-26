from abc import ABC, abstractmethod
import torch.nn as nn

class BaseICLModel(ABC, nn.Module):
    """
    Абстрактный базовый класс для всех In-Context Learning моделей.

    Наследуется от ABC и nn.Module

    """

    @abstractmethod
    def forward(self, x):
        """
        Прямой проход модели через все слои.
        
        Args:
            tokens: Токенизированная последовательность формы (batch, seq_len, dim).
            
        Returns:
            torch.Tensor: Преобразованные токены той же формы (batch, seq_len, dim).
        """
        return NotImplementedError

    @abstractmethod
    def predict(self, tokens):
        """
        Возвращает предсказание модели для тестового токена.
        
        Извлекает y-часть последнего токена в последовательности и применяет
        необходимые преобразования.
        
        Args:
            tokens: Токенизированная последовательность формы (batch, seq_len, dim).
            
        Returns:
            torch.Tensor: Предсказанные значения y формы (batch, ny).
        """
        return NotImplementedError