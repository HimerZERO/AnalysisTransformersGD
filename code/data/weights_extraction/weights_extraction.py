from sklearn.linear_model import LinearRegression
from data.linear import LinearTaskGenerator, TaskGenerator
import torch
import numpy as np

def weight_extraction(model,
                      device,
                      generator: TaskGenerator,
                      samples_num: int = 1000,
                      N: int = 100,
                      refresh_generator: bool = False,
                      verbose: bool = True
                      ):
    """
    TODO

    Args:
        model: модель, реализующая методы forward и predict
        task_generator: Генератор задач с методом .generate_task(N).
        N: Количество обучающих примеров в контексте.
        NX, NY: Размерности задачи
        verbose: Если True, показывает прогресс-бар и графики.

    Returns:
        Matrix: Наилучше приближающая модель матрица
        R^2: R^2-score матрицы

    targets = np.array(targets)
    """
    NX: int = generator.nx
    NY: int = generator.ny
    tokens = []
    assert samples_num > 20 * (NX + NY), "Количество сэмплов должно быть сильно больше размерности векторов"

    if refresh_generator:
        generator.generate_teacher()

    for _ in range(samples_num):
        x, _ = generator.generate_task(N=N, new_params=False)
        if len(tokens) != 0:
            tokens.append(tokens[-1].copy())
            tokens[-1][N] = x[N]
            continue
        tokens.append(x)

    tokens = torch.FloatTensor(np.array(tokens)).to(device)
    tokens_ends = tokens.detach().numpy()[:, N, :NX]

    model.eval()
    pred = model.predict(tokens).detach().numpy()

    W_true = generator.teacher_params
    lin_reg = LinearRegression(fit_intercept=False).fit(tokens_ends, pred)

    if verbose:
        print("||W_true - W_extr||_2 =", np.linalg.norm(W_true - lin_reg.coef_))
        print("R^2 =", lin_reg.score(tokens_ends, pred))
    return lin_reg.coef_, lin_reg.score(tokens_ends, pred)