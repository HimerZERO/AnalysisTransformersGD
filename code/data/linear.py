from .base import TaskGenerator
import numpy as np

class LinearTaskGenerator(TaskGenerator):
    """
    Генератор задач линейной регрессии для In-Context Learning.
    
    Генерирует задачи, в которых истинная зависимость y = W @ x,
    где W - случайная матрица весов из нормального распределения. 
    Наследуется от TaskGenerator

    Attributes:
        nx (int): Размерность фичей
        ny (int): Размерность таргета. По умолчанию 1.
        x_range (tuple[float, float]): Границы равномерного распределения для генерации признаков x.
        w_std (float): стандартное отклонение распределения, из которого генерируются матрицы.
    """

    def __init__(self, nx: int, ny: int, x_range: tuple[float, float], w_std: float = 1.0):
        """
        Инициализирует генератор задач линейной регрессии.
        
        Args:
            nx: Размерность признаков x.
            ny: Размерность целевой переменной y.
            x_range: Кортеж (min, max) для равномерной генерации x.
            w_std: Стандартное отклонение весов учителя.
        """

        super().__init__(nx, ny, x_range)
        self.w_std = w_std
    
    def generate_teacher(self):
        """
        Генерирует матрицу весов учителя из нормального распределения.
        
        Returns:
            W: np.ndarray формы (ny, nx) - матрица весов W.
        """

        return np.random.normal(loc=0, scale=self.w_std, size=(self.ny, self.nx))
    
    def compute_y(self, teacher_params, X):
        """
        Вычисляет значения y = W @ X^T для заданных X.
        
        Args:
            teacher_params: Матрица весов W формы (ny, nx).
            X: Матрица признаков формы (n_samples, nx).
            
        Returns:
            Y: np.ndarray формы (n_samples, ny) - вычисленные значения y.
        """
        return X @ teacher_params.T


    