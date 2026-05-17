from abc import ABC, abstractmethod
import numpy as np

class TaskGenerator(ABC):
    """
    Абстрактный генератор задач для In-Context Learning.

    Генерирует батчи задач, каждая из которых состоит из контекста
    (N пар x-y) и одного тестового запроса. Данные создаются на основе
    случайного "учителя", параметры которого определяются в наследниках.

    Attributes:
        nx (int): Размерность признаков x.
        ny (int): Размерность целевой переменной y.
        x_range (tuple[float, float]): Границы равномерного распределения
            для генерации признаков x.
        noise (float | None): Стандартное отклонение гауссовского шума
            в контексте. None - без шума.
    """

    def __init__(self, nx: int, ny: int = 1, x_range: tuple = (-1, 1), noise=None):
        """
        Инициализирует генератор задач.

        Args:
            nx: Размерность признаков x.
            ny: Размерность целевой переменной y.
            x_range: Кортеж (min, max) для равномерной генерации x.
            noise: Стандартное отклонение гауссовского шума в контексте. None - без шума.
        """
        
        self.nx = nx
        self.ny = ny
        self.x_range = x_range
        self.noise = noise
  
    @abstractmethod
    def generate_teacher(self, batch_size):
        """
        Генерирует параметры учителей для батча задач.

        Args:
            batch_size: Количество учителей (размер батча).

        Returns:
            Параметры учителей. Для линейной модели - np.ndarray формы (batch_size, ny, nx).
        """
        pass

    @abstractmethod
    def compute_y(self, teacher_params, X: np.ndarray) -> np.ndarray:
        """
        Вычисляет значения y для заданных x по параметрам учителей.

        Args:
            teacher_params: Параметры учителей, возвращённые
                generate_teacher(batch_size).
            X: Матрица признаков формы (batch_size, N, nx).

        Returns:
            Матрица целевых значений формы (batch_size, N, ny).
        """

        pass
    
    @abstractmethod
    def generate_batch(self, batch_size: int, N: int, new_params: bool = True):
        """
        Генерирует батч задач.

        Args:
            batch_size: Количество задач в батче.
            N: Количество обучающих примеров в контексте.
            new_params: Если True, каждый пример в батче получает
                своего учителя. Если False, весь батч использует
                одного учителя self.teacher_params.

        Returns:
            tokens: torch.Tensor формы (batch_size, N+1, nx+ny).
            y_test_true: torch.Tensor формы (batch_size, ny) - правильный ответ для теста (всегда чистый, без шума).
        """

        pass

    def _tokenize(self, X_context: np.ndarray, Y_context: np.ndarray, X_test: np.ndarray, W0: np.ndarray | None = None) -> np.ndarray:
        """
        Формирует последовательность токенов для подачи в модель.

        Контекстные токены имеют вид [x_i, y_i].
        Тестовый токен имеет вид [x_test, -W0 @ x_test].
        При W0=None используется нулевая матрица.

        Args:
            X_context: Матрица контекстных признаков (batch_size, N, nx).
            Y_context: Матрица контекстных ответов (batch_size, N, ny).
            X_test: Матрица тестовых признаков (batch_size, 1, nx).
            W0: Начальная матрица весов (ny, nx). При None - нулевая.

        Returns:
            np.ndarray формы (batch_size, N+1, nx+ny).
        """

        if W0 is None:
            W0 = np.zeros((self.ny, self.nx))

        context_tokens = np.concatenate([X_context, Y_context], axis=-1)
        initial_y = -(X_test @ W0.T)
        test_token = np.concatenate([X_test, initial_y], axis=-1)

        return np.concatenate([context_tokens, test_token], axis=1)