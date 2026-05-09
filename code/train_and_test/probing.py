from sklearn.linear_model import LinearRegression
from data.base import TaskGenerator
import torch
import numpy as np

def weight_extraction(
        model: torch.nn.Module,
        generator: TaskGenerator,
        device: str = 'cpu',
        samples_num: int = 1000,
        N: int = 10,
        refresh_generator: bool = False
    ) -> torch.Tensor:
    """
    Извлекает эффективную матрицу весов обученной модели методом OLS

    Args:
        model:             Модель, реализующая методы forward и predict
        task_generator:    Генератор задач с методом .generate_batch(). TODO Заработает после изменения файла генераторов
        device:            Устройство для вычислений.
        samples_num:       Количество точек для восстановления матрицы.
        N:                 Количество обучающих примеров в контексте.
        refresh_generator: Если True, создаёт новую задачу генератора.

    Returns:
        Matrix: Наилучше приближающая модель матрица

    targets = np.array(targets)
    """
    if refresh_generator:
        generator.generate_teacher()

    NX: int = generator.nx
    NY: int = generator.ny
    assert samples_num > 20 * (NX + NY), "Количество сэмплов должно быть сильно больше размерности векторов"

    tokens_np, _ = generator.generate_batch(batch_size=samples_num, N=N)
    tokens = torch.FloatTensor(tokens_np).to(device)
    tokens_ends = tokens_np[:, N, :NX]

    model.eval()
    pred = model.predict(tokens).cpu().numpy()

    lin_reg = LinearRegression(fit_intercept=False).fit(tokens_ends, pred)

    theta_hat = lin_reg.coef_
    theta_hat = theta_hat.reshape(NY, NX)

    return torch.FloatTensor(theta_hat)
