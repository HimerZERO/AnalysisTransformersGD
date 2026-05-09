import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any

def matrix_cosine_similarity(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    Вычисляет косинусное сходство между двумя матрицами
    """
    return F.cosine_similarity(A.flatten(), B.flatten(), dim=0).item()

def matrix_l2_distance(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    Вычисляет евклидово расстояние между матрицами.
    """
    return torch.norm(A - B, p='fro').item()

def trajectory_proximity(theta_hat: torch.Tensor, trajectory: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
    """
    Находит ближайший шаг в траектории GD к извлеченной матрице весов.

    Аргументы:
      theta_hat:  Извлеченная матрица (ny, nx).
      trajectory: Траектория весов GD (T, ny, nx)

    Возвращает: (номер шага, расстояние, матрица на этом шаге)
    """
    distances = [matrix_l2_distance(theta_hat, gd_mtx) for gd_mtx in trajectory]

    distances = torch.tensor(distances)
    min_dist, closest_step = torch.min(distances, dim=0)

    step_idx = closest_step.item()
    return step_idx, min_dist.item(), trajectory[step_idx]

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Вычисляет коэффициент детерминации.
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)

    r2 = 1 - ss_res / ss_tot
    return r2.item()

def evaluate_model(
        theta_hat: torch.Tensor,
        trajectory: torch.Tensor,
        X_test: torch.Tensor,
        y_pred: torch.Tensor,
        y_true_clean: torch.Tensor) -> Dict[str, Any]:
    """
    Проводит комплексную оценку модели.

    Аргументы:
      theta_hat:    Извлеченная матрица
      trajectory:   Траектория весов GD
      X_test:       Тестовые входы
      y_pred:       Предсказания модели
      y_true_clean: Истинные значения без шума

    Возвращает:
      r2_score (Коэфициент детерминации, насколько линейно обучилась)

      closest_step      (Номер шага)
      l2_distance       (Дистанция шага)
      theta_at_step     (Матрица шага)
      cosine_similarity (косинусное расстояние)

      clean_target_mse (MSE от истинного)
    """
    y_pred_linear = X_test @ theta_hat.T
    r2 = r2_score(y_pred, y_pred_linear)

    step_idx, dist_traj, theta_gd = trajectory_proximity(theta_hat, trajectory)

    cos_sim = matrix_cosine_similarity(theta_hat, theta_gd)

    mse_clean = F.mse_loss(y_pred, y_true_clean).item()

    return {
        'r2_score': r2,
        'closest_step': step_idx,
        'l2_distance': dist_traj,
        'theta_at_step': theta_gd,
        'cosine_similarity': cos_sim,
        'clean_target_mse': mse_clean
    }
