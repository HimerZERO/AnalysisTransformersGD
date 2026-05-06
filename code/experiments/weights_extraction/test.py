import torch
import numpy as np

def get_SignleLSA_matrices(model) -> tuple:
    """
    Извлекает матрицы весов из однослойной LSA модели.
    
    Args:
        model: Обученная модель SingleLayerLSA.
    
    Returns:
        tuple: Кортеж из четырёх матриц (W_q, W_k, W_v, P).
               Каждая матрица имеет форму (dim, dim).
    """

    layer = model.layer
    W_q = layer.W_q.weight.data.clone()
    W_k = layer.W_k.weight.data.clone()
    W_v = layer.W_v.weight.data.clone()
    P = layer.P.weight.data.clone()
    return W_q, W_k, W_v, P

def compare_matrices(target_M: torch.Tensor, true_M: torch.Tensor) -> dict:
    """
    Сравнивает две матрицы по нескольким метрикам.
    
    Args:
        target_M: Целевая матрица (напр., из обученной модели).
        true_M: Эталонная матрица (напр., теоретическая GD).
    
    Returns:
        dict: Словарь с метриками:
            - 'cosine_similarity': Косинусное сходство (от -1 до 1).
            - 'l2_distance': Евклидово расстояние между матрицами.
            - 'relative_error': Относительная ошибка (l2_dist / norm(true_M)).
    """

    cos_sim = torch.nn.functional.cosine_similarity(target_M.flatten(), true_M.flatten(), dim=0).item()
    l2_dist = torch.norm(target_M - true_M).item()
    norm = torch.norm(true_M).item()
    rel_error = l2_dist / (norm + 1e-8)
    
    return {
        'cosine_similarity': cos_sim,
        'l2_distance': l2_dist,
        'relative_error': rel_error
    }

def get_theoretical_gd_matrices(nx: int, ny: int, eta: float, N: int, W0: torch.Tensor = None) -> tuple:
    """
    Создаёт теоретические матрицы для одного шага градиентного спуска.
    
    Матрицы:
        W_q[:nx, :nx] = I
        W_k[:nx, :nx] = I
        W_v[:nx, nx:] = -I
        P[nx:, :nx] = (eta / N) * I
    
    Args:
        nx: Размерность признаков x.
        ny: Размерность целевой переменной y.
        eta: Learning rate градиентного спуска.
        N: Количество обучающих примеров в контексте.
        W0: Начальные веса модели формы (ny, nx). Если None, используется 0.
    
    Returns:
        tuple: Кортеж из четырёх матриц (W_q, W_k, W_v, P).
    """
    
    if W0 is None:
        W0 = torch.zeros(ny, nx)
    
    d = nx + ny
    
    W_q = torch.zeros(d, d)
    W_q[:nx, :nx] = torch.eye(nx)
    
    W_k = torch.zeros(d, d)
    W_k[:nx, :nx] = torch.eye(nx)
    
    W_v = torch.zeros(d, d)
    W_v[:nx, nx:] = -torch.eye(nx, ny)
    
    P = torch.zeros(d, d)
    P[nx:, :nx] = (eta / N) * torch.eye(ny, nx)
    
    return W_q, W_k, W_v, P


def compare_SignelLSA_matrices(model, nx: int, ny: int, eta: float, N: int, W0: torch.Tensor = None, device: str = 'cpu') -> dict:
    """
    Сравнивает матрицы обученной модели с теоретическими GD-матрицами.
    
    Извлекает веса из модели, создаёт теоретические матрицы для заданной eta,
    и вычисляет метрики сходства для каждой из четырёх матриц (W_q, W_k, W_v, P).
    
    Args:
        model: Обученная модель SingleLayerLSA.
        nx: Размерность признаков x.
        ny: Размерность целевой переменной y.
        eta: Learning rate для теоретических GD-матриц.
        N: Количество обучающих примеров в контексте.
        W0: Начальные веса модели. Если None, используется 0.
        device: Устройство для вычислений ('cpu' или 'cuda').
    
    Returns:
        dict: Словарь с результатами сравнения:
            - 'W_q', 'W_k', 'W_v', 'P': Метрики для каждой матрицы.
            - 'mean': Средние значения метрик по всем четырём матрицам.
            
    """

    model = model.to(device)
    
    W_q_model, W_k_model, W_v_model, P_model = get_SignleLSA_matrices(model)
    W_q_theory, W_k_theory, W_v_theory, P_theory = get_theoretical_gd_matrices(nx, ny, eta, N, W0)
    
    W_q_theory = W_q_theory.to(device)
    W_k_theory = W_k_theory.to(device)
    W_v_theory = W_v_theory.to(device)
    P_theory = P_theory.to(device)
    
    results = {
        'W_q': compare_matrices(W_q_model, W_q_theory),
        'W_k': compare_matrices(W_k_model, W_k_theory),
        'W_v': compare_matrices(W_v_model, W_v_theory),
        'P': compare_matrices(P_model, P_theory),
    }
    
    results['mean'] = {
        'cosine_similarity': np.mean([results[k]['cosine_similarity'] for k in ['W_q', 'W_k', 'W_v', 'P']]),
        'l2_distance': np.mean([results[k]['l2_distance'] for k in ['W_q', 'W_k', 'W_v', 'P']]),
        'relative_error': np.mean([results[k]['relative_error'] for k in ['W_q', 'W_k', 'W_v', 'P']])
    }
    
    return results