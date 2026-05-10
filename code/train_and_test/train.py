import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from train_and_test.gradient_descent import gd_trajectory

def train_ICL_model(model, task_generator, 
                    optimizer,
                    n_steps: int = 2000, 
                    batch_size: int = 2048,
                    device='cpu', 
                    N: int = 10,
                    verbose: bool = True, 
                    test_interval: int = 100,
                    val_tasks: int = 1000,
                    lr_values: list[float] = None,
                    gd_loss_type: str = 'mse',
                    huber_delta: float = 1.0,
                    gd_n_steps: int = None):
    '''
    Обучает In-Context Learning модель на задачах из генератора.
    
    Args:
        model: модель, реализующая методы forward и predict
        task_generator: Генератор задач с методом .generate_task(N).
        n_steps: Количество шагов обучения.
        batch_size: Размер батча.
        N: Количество обучающих примеров в контексте.
        device: Устройство для вычислений.
        verbose: Если True, показывает прогресс-бар и графики.
        test_interval: Интервал для тестирования и сравнения с GD.
        val_tasks: Количество задач для валидации.
        lr_values: Список lr для подбора оптимальной GD.
        gd_loss_type: Тип функции потерь для GD - 'mse' или 'huber'.
            По умолчанию 'mse'.
        huber_delta: Параметр дельта для Huber-loss. По умолчанию 1.0.
        gd_n_steps: Количество шагов GD. По умолчанию model.n_layers.
    
    Returns:
        model: Обученная модель.
        losses: Список train loss.
        tf_val_losses: Список valid loss модели.
        gd_val_losses: Список valid loss модели GD.
        best_lr: Оптимальная eta для GD.
    '''

    if lr_values is None:
        lr_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    if gd_n_steps is None:
        gd_n_steps = model.n_layers
    
    model = model.to(device)
    loss_fn = nn.MSELoss()
    losses = []
    tf_val_losses = []
    val_steps = []
    
    val_tokens, val_targets = task_generator.generate_batch(batch_size=val_tasks, N=N)
    val_tokens = val_tokens.to(device)
    val_targets = val_targets.to(device) 

    X_val = val_tokens[:, :-1, :model.nx]    # (val_tasks, N, nx)
    y_val = val_tokens[:, :-1, model.nx:]    # (val_tasks, N, ny)
    X_query = val_tokens[:, -1:, :model.nx]  # (val_tasks, 1, nx)
    
    best_lr = None
    best_gd_loss = float('inf')
    for gd_lr in tqdm(lr_values):
        total_loss = 0.0
        for i in range(val_tasks):
            X_i = X_val[i]
            y_i = y_val[i]

            traj = gd_trajectory(
                X_i, y_i,
                n_steps=gd_n_steps,
                lr=gd_lr,
                W0=None,
                loss_type=gd_loss_type,
                huber_delta=huber_delta
            )
            W_final = traj[-1]

            x_query = X_query[i]
            y_pred_gd = x_query @ W_final.T

            total_loss += loss_fn(y_pred_gd.squeeze(0), val_targets[i]).item()

        avg_loss = total_loss / val_tasks

        if avg_loss < best_gd_loss:
            best_gd_loss = avg_loss
            best_lr = gd_lr
    
    steps = range(n_steps)
    iterator = tqdm(steps, desc='Training') if verbose else steps
    
    for step in iterator:
        model.train()

        batch_tokens, batch_targets = task_generator.generate_batch(batch_size=batch_size, N=N)
        tokens = batch_tokens.to(device)
        targets = batch_targets.to(device)

        pred = model.predict(tokens)
        loss = loss_fn(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001)
        optimizer.step()

        losses.append(loss.item())

        if step % test_interval == 0:
            model.eval()
            with torch.no_grad():
                pred_tf = model.predict(val_tokens)
                loss_tf = loss_fn(pred_tf, val_targets).item()
                
                tf_val_losses.append(loss_tf)
                val_steps.append(step)
            
            if verbose:
                if gd_loss_type == 'mse':
                    gd_label = f'GD MSE (lr = {best_lr:.4f})'
                else:
                    gd_label = f'GD Huber (lr = {best_lr:.4f}, $\Delta$ = {huber_delta:.2f})'

                iterator.set_postfix(
                    train_loss=loss.item(),
                    tf_val=f'{loss_tf:.4f}',
                    gd_val=f'{best_gd_loss:.4f}'
                )

                clear_output(wait=True)
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                axes[0].plot(losses, linewidth=1, color='#0000FF')
                axes[0].set_xlabel('Step')
                axes[0].set_ylabel('MSE Loss')
                axes[0].set_title('Training Loss')
                axes[0].set_yscale('log')
                axes[0].grid(True, alpha=0.3)

                tf_min = min(tf_val_losses) if tf_val_losses else 0
                tf_max = max(tf_val_losses) if tf_val_losses else 1

                gd_is_visible = (best_gd_loss > tf_min * 0.1)

                if gd_is_visible:
                    axes[1].set_yscale('linear')
                    y_min = min(tf_min, best_gd_loss) * 0.9
                    y_max = max(tf_max, best_gd_loss) * 1.1
                else:
                    axes[1].set_yscale('log')
                    y_min = None
                    y_max = None

                axes[1].plot(val_steps, tf_val_losses, 'o-', color='#3366CC',
                            label='Trained TF', markersize=3, lw=1.5)
                axes[1].axhline(y=best_gd_loss, color='#FF0000', linestyle='--', lw=2,
                                label=gd_label)
                axes[1].set_xlabel('Step')
                axes[1].set_ylabel('MSE Loss')
                axes[1].set_title('Validation: TF vs GD')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                if not gd_is_visible:
                    axes[1].text(0.98, 0.02, f'GD loss: {best_gd_loss:.6f}',
                                transform=axes[1].transAxes,
                                ha='right', va='bottom',
                                fontsize=9, color='#FF0000',
                                bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='white', alpha=0.8))

                if y_min is not None and y_max is not None:
                    axes[1].set_ylim(y_min, y_max)

                plt.tight_layout()
                plt.show()

    return model, losses, tf_val_losses, best_lr
