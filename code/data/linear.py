from .base import TaskGenerator
import numpy as np
import torch

class LinearTaskGenerator(TaskGenerator):
    """
    Генератор задач линейной регрессии для In-Context Learning.

    Генерирует задачи, в которых истинная зависимость y = W @ x,
    где W - случайная матрица весов из нормального распределения.

    Attributes:
        nx (int): Размерность признаков x.
        ny (int): Размерность целевой переменной y.
        x_range (tuple[float, float]): Границы равномерного распределения
            для генерации признаков x.
        noise (float | None): Стандартное отклонение гауссовского шума
            в контексте. None - без шума.
        w_std (float): Стандартное отклонение весов учителя.
    """

    def __init__(self, nx: int, ny: int = 1, x_range: tuple = (-1, 1), noise: float | None = None, w_std: float = 1.0):
        """
        Инициализирует генератор задач линейной регрессии.

        Args:
            nx: Размерность признаков x.
            ny: Размерность целевой переменной y.
            x_range: Кортеж (min, max) для равномерной генерации x.
            noise: Стандартное отклонение гауссовского шума в контексте.
                None — без шума.
            w_std: Стандартное отклонение весов учителя.
        """

        super().__init__(nx, ny, x_range, noise)
        self.w_std = w_std
    
    def generate_teacher(self, batch_size):
        """
        Генерирует батч матриц весов из нормального распределения N(0, w_std).

        Сохраняет первый элемент в self.teacher_params для повторного
        использования с new_params=False.

        Args:
            batch_size: Количество матриц (размер батча).

        Returns:
            np.ndarray формы (batch_size, ny, nx).
        """

        W = np.random.normal(0, self.w_std, size=(batch_size, self.ny, self.nx))

        self.teacher_params = W[0]
        return W
    
    def compute_y(self, W, X):
        """
        Вычисляет y = X @ W^T с аддитивным гауссовским шумом.

        Args:
            W: Матрицы весов (batch_size, ny, nx).
            X: Матрица признаков (batch_size, N, nx).

        Returns:
            np.ndarray формы (batch_size, N, ny).
        """

        Y = X @ W.transpose(0, 2, 1)
        if self.noise is not None:
            Y += np.random.normal(0, self.noise, size=Y.shape)
        return Y
    
    def generate_batch(self, batch_size: int, N: int, new_params: bool = True):
        if new_params:
            W = self.generate_teacher(batch_size)
        else:
            W = self.teacher_params[np.newaxis, ...].repeat(batch_size, axis=0)

        X_ctx = np.random.uniform(*self.x_range,
                                  size=(batch_size, N, self.nx))
        Y_ctx = self.compute_y(W, X_ctx)

        X_test = np.random.uniform(*self.x_range,
                                   size=(batch_size, 1, self.nx))
        Y_test = X_test @ W.transpose(0, 2, 1)  # чистый

        tokens = self._tokenize(X_ctx, Y_ctx, X_test)
        return torch.FloatTensor(tokens), torch.FloatTensor(Y_test.squeeze(1))


class OutlierLinearGenerator(LinearTaskGenerator):
    """
    Генератор задач линейной регрессии с выбросами.

    Добавляет к контекстным y выбросы большой амплитуды.
    Тестовый ответ всегда остаётся чистым.

    Attributes:
        nx (int): Размерность признаков x.
        ny (int): Размерность целевой переменной y.
        x_range (tuple[float, float]): Границы равномерного распределения для генерации признаков x.
        noise (float | None): Стандартное отклонение гауссовского шума.
        w_std (float): Стандартное отклонение весов учителя.
        outlier_prob (float): Вероятность выброса (от 0 до 1).
        outlier_scale (float): Амплитуда выброса.
        outlier_direction (str): Направление выбросов - 'both', 'left' или 'right'.
    """

    def __init__(self, nx: int, ny: int = 1,
                 x_range: tuple = (-1, 1),
                 noise: float | None = None,
                 w_std: float = 1.0,
                 outlier_prob: float = 0.1,
                 outlier_scale: float = 10.0,
                 outlier_direction: str = 'both'):
        """
        Инициализирует генератор задач с выбросами.

        Args:
            nx: Размерность признаков x.
            ny: Размерность целевой переменной y.
            x_range: Кортеж (min, max) для равномерной генерации x.
            noise: Стандартное отклонение гауссовского шума.
            w_std: Стандартное отклонение весов учителя.
            outlier_prob: Вероятность выброса (от 0 до 1).
            outlier_scale: Амплитуда выброса.
            outlier_direction: 'both', 'left' или 'right'.
        """
        super().__init__(nx, ny, x_range, noise, w_std)
        self.outlier_prob = outlier_prob
        self.outlier_scale = outlier_scale
        self.outlier_direction = outlier_direction

    def _add_outliers(self, Y: np.ndarray) -> np.ndarray:
        """
        Добавляет выбросы в батч целевых переменных.

        Args:
            Y: Матрица целевых значений (batch_size, N, ny).

        Returns:
            np.ndarray той же формы с добавленными выбросами.
        """

        batch_size, N, ny = Y.shape
        mask = np.random.random(size=(batch_size, N, 1)) < self.outlier_prob

        if self.outlier_direction == 'both':
            signs = np.random.choice([-1, 1], size=(batch_size, N, 1))
        elif self.outlier_direction == 'left':
            signs = -np.ones((batch_size, N, 1))
        elif self.outlier_direction == 'right':
            signs = np.ones((batch_size, N, 1))
        else:
            raise ValueError(...)

        return Y + mask * signs * self.outlier_scale
    

    def compute_y(self, teacher_params, X):
        """
        Вычисляет y = X @ W^T с шумом и выбросами.

        Args:
            teacher_params: Матрица весов W формы (ny, nx).
            X: Матрица признаков (n_samples, nx) или
                (batch_size, N, nx) для батча.

        Returns:
            np.ndarray с добавленным шумом и выбросами.
        """
        Y = super().compute_y(teacher_params, X)
        return self._add_outliers(Y)
    