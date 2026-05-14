import torch

def gd_step(X, y, W, lr, loss_type='mse', huber_delta=1.0) -> torch.Tensor:
    """
    Выполняет один шаг градиентного спуска по выбранной функции потерь.

    Args:
        X (torch.Tensor): Матрица фичей размера (N, nx), где N - число примеров,
            nx - размерность признаков.
        y (torch.Tensor): Матрица целевых переменных размера (N, ny), где
            ny - размерность ответа.
        W (torch.Tensor): Текущая матрица весов размера (ny, nx).
        lr (float): Скорость обучения (learning rate).
        loss_type (str): Тип функции потерь - 'mse' или 'huber'.
            По умолчанию 'mse'.
        huber_delta (float): Параметр дельта для Huber-loss. Используется
            только при loss_type='huber'. По умолчанию 1.0.

    Returns:
        torch.Tensor: Обновлённая матрица весов размера (ny, nx).
    """

    y_pred = X @ W.T

    residuals = y_pred - y

    if loss_type == 'mse':
        grad = (2.0 / X.shape[0]) * (residuals.T @ X)
    
    elif loss_type == 'huber':
        grad_residuals = torch.clamp(residuals, min=-huber_delta, max=huber_delta)
        grad = (grad_residuals.T @ X) / X.shape[0]

    
    W_new = W - lr * grad

    return W_new

def gd_trajectory(X, y, n_steps, lr, W0=None, loss_type='mse', huber_delta=1.0):
    """
    Вычисляет траекторию весов на протяжении n_steps шагов градиентного спуска.

    Args:
        X (torch.Tensor): Матрица фичей размера (N, nx), где N - число примеров,
            nx - размерность признаков.
        y (torch.Tensor): Матрица целевых переменных размера (N, ny), где
            ny - размерность ответа.
        n_steps (int): Количество шагов градиентного спуска.
        lr (float): Скорость обучения (learning rate).
        W0 (torch.Tensor | None): Начальная матрица весов размера (ny, nx).
            При None инициализируется нулевой. По умолчанию None.
        loss_type (str): Тип функции потерь - 'mse' или 'huber'.
            По умолчанию 'mse'.
        huber_delta (float): Параметр δ для Huber-loss. Используется
            только при loss_type='huber'. По умолчанию 1.0.

    Returns:
        torch.Tensor: Траектория весов размера (n_steps + 1, ny, nx).
            Индекс 0 - начальные веса (W0), индекс k - веса после k шагов GD.
    """
    
    nx = X.shape[1]
    ny = y.shape[1]

    if W0 is None:
        W = torch.zeros(ny, nx, dtype=X.dtype, device=X.device)
    else:
        W = W0.clone()

    trajectory = [W.clone()]

    for _ in range(n_steps):
        W = gd_step(X, y, W, lr, loss_type, huber_delta)
        trajectory.append(W.clone())

    return torch.stack(trajectory, dim=0)