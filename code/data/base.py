from abc import ABC, abstractmethod
import numpy as np

class TaskGenerator(ABC):
    """
    Представляет генератор задачи для модели в рамках проекта

    Генерирует задачи, состоящие из контекста (N пар x-y) и тестового запроса.
    Данные создаются на основе случайного "учителя", параметры которого
    определяются в наследниках. Наследуется от ABC

    Attributes:
        nx (int): Размерность фичей
        ny (int): Размерность таргета.
        x_range (tuple[float, float]): Границы равномерного распределения для генерации признаков x.
    """

    def __init__(self, nx: int, ny: int = 1, x_range: tuple = (-1, 1), noise=None):
        """
        Инициализирует генератор задач.

        Args:
            nx: Размерность признаков x.
            ny: Размерность целевой переменной y.
            x_range: Кортеж (min, max) для равномерной генерации x.
            noise: Стандартное отклонение для генерации шума из нормального распределения. None для генерации без шума
        """
        
        self.nx = nx
        self.ny = ny
        self.x_range = x_range
        self.noise = noise
  
    @abstractmethod
    def generate_teacher(self):
        """
        Генерирует параметры скрытого "учителя".

        Учитель определяет истинную зависимость y = f(x).

        Returns:
            Любой объект, содержащий параметры учителя (например, матрица весов).
        """
        pass

    @abstractmethod
    def compute_y(self, teacher_params, X: np.ndarray) -> np.ndarray:
        """
        Вычисляет значения y для заданных x по параметрам учителя.

        Args:
            teacher_params (any): Параметры учителя, возвращённые generate_teacher().
            X (np.array): Матрица признаков формы (N, nx).

        Returns:
            Матрица целевых значений формы (n_samples, ny).
        """

        pass

    def generate_task(self, N: int, new_params: bool = True):
        """
        Генерирует одну задачу для In-Context Learning.
        
        Задача состоит из N обучающих примеров в контексте и одного тестового
        запроса. Возвращает токенизированную последовательность и истинный
        ответ для тестового запроса.

        Args:
            N (int): Количество обучающих примеров в контексте.

        Returns:
            tokens: np.ndarray формы (N+1, nx+ny) - последовательность токенов.
            Y_test_true: np.ndarray формы (ny,) — правильный ответ для теста.
        """

        teacher = self.generate_teacher() if new_params else self.teacher_params

        X_context = np.random.uniform(*self.x_range, size=(N, self.nx))
        Y_context = self.compute_y(teacher, X_context)

        X_test = np.random.uniform(*self.x_range, size=(1, self.nx))
        Y_test_true = self.compute_y(teacher, X_test)

        tokens = self._tokenize(X_context, Y_context, X_test)

        return tokens, Y_test_true.flatten()

    def _tokenize(self, X_context: np.ndarray, Y_context: np.ndarray, X_test: np.ndarray, W0: np.ndarray | None = None) -> np.ndarray:
        """
        Формирует последовательность токенов для подачи в модель

        Обучающие токены имеют вид [x_i, y_i]. Тестовый токен имеет вид
        [x_test, -W0 @ x_test]. При W0=None используется нулевая матрица.

        Args:
            X_context (np.array): Матрица контекстных признаков формы (N, nx).
            Y_context (np.array): Матрица контекстных таргетов формы (N, ny).
            X_test (np.array): Матрица тестовых признаков формы (1, nx).
            W0 (np.array): Начальная матрица весов модели формы (ny, nx). Если None, используется нулевая матрица.

        Returns:
            Матрица токенов формы (N+1, nx+ny), где первые nx столбцов признаки, оставшиеся ny - таргеты.
        """

        if W0 is None:
            W0 = np.zeros((self.ny, self.nx))

        context_tokens = np.concatenate([X_context, Y_context], axis=1)

        initial_y = -(W0 @ X_test.T).T.flatten()
        test_token = np.concatenate([X_test.flatten(), initial_y])

        return np.vstack([context_tokens, test_token])