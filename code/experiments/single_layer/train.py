import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from models.linear_attention import SingleLayerLSA

def train_ICL_model(model, task_generator, 
                    optimizer,
                    n_steps: int = 2000, 
                    batch_size: int = 2048, 
                    lr: float = 0.001, N: int = 10, 
                    device='cpu', 
                    verbose: bool = True, 
                    test_interval: int = 100,
                    val_tasks: int = 1000,
                    eta_values: list[float] = None):
    '''
    Обучает In-Context Learning модель на задачах из генератора.
    
    Периодически сравнивает loss модели с loss градиентного спуска
    (через LSA с GD-матрицами).
    
    Args:
        model: модель, реализующая методы forward и predict
        task_generator: Генератор задач с методом .generate_task(N).
        n_steps: Количество шагов обучения.
        batch_size: Размер батча.
        lr: Learning rate для оптимизатора.
        N: Количество обучающих примеров в контексте.
        device: Устройство для вычислений.
        verbose: Если True, показывает прогресс-бар и графики.
        test_interval: Интервал для тестирования и сравнения с GD.
        val_tasks: Количество задач для валидации.
        eta_values: Список eta для подбора оптимальной GD.
    
    Returns:
        model: Обученная модель.
        losses: Список train loss.
        tf_val_losses: Список valid loss модели.
        gd_val_losses: Список valid loss модели GD.
        best_eta: Оптимальная eta для GD.
    '''

    if eta_values is None:
        eta_values = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    
    model = model.to(device)
    loss_fn = nn.MSELoss()
    losses = []
    tf_val_losses = []
    gd_val_losses = []
    val_steps = []
    
    gd_model = SingleLayerLSA(nx=model.nx, ny=model.ny).to(device) # это GD, реализованное эталонными матрицами в SingleLayerLSA
    
    val_tokens = []
    val_targets = []
    for _ in range(val_tasks):
        tokens, y_true = task_generator.generate_task(N=N)
        val_tokens.append(tokens)
        val_targets.append(y_true)
    
    val_tokens = torch.FloatTensor(np.array(val_tokens)).to(device)
    val_targets = torch.FloatTensor(np.array(val_targets)).to(device)
    
    best_eta = None
    best_gd_loss = float('inf')
    for eta in eta_values:
        gd_model.set_weights_from_gd(eta=eta, N=N)
        with torch.no_grad():
            pred_gd = gd_model.predict(val_tokens)
            loss_gd = loss_fn(pred_gd, val_targets).item()
        if loss_gd < best_gd_loss:
            best_gd_loss = loss_gd
            best_eta = eta
    
    gd_model.set_weights_from_gd(eta=best_eta, N=N) # Нашли лучшую eta, теперь работае только с ней
    
    steps = range(n_steps)
    iterator = tqdm(steps, desc='Training') if verbose else steps
    
    for step in iterator:
        model.train()
        batch_tokens = []
        batch_targets = []

        for _ in range(batch_size):
            tokens, y_true = task_generator.generate_task(N=N)
            batch_tokens.append(tokens)
            batch_targets.append(y_true)
        
        tokens = torch.FloatTensor(np.array(batch_tokens)).to(device)
        targets = torch.FloatTensor(np.array(batch_targets)).to(device)

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
                
                pred_gd = gd_model.predict(val_tokens)
                loss_gd = loss_fn(pred_gd, val_targets).item()
                
                tf_val_losses.append(loss_tf)
                gd_val_losses.append(loss_gd)
                val_steps.append(step)
            
            if verbose:
                iterator.set_postfix(
                    train_loss=loss.item(),
                    tf_val=f'{loss_tf:.4f}',
                    gd_val=f'{loss_gd:.4f}'
                )
                
                clear_output(wait=True)
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                axes[0].plot(losses, linewidth=1, color='#0000FF')
                axes[0].set_xlabel('Step')
                axes[0].set_ylabel('MSE Loss')
                axes[0].set_title('Training Loss')
                axes[0].set_yscale('log')
                axes[0].grid(True, alpha=1)
                
                axes[1].plot(val_steps, tf_val_losses, 'o-', label='Trained TF', markersize=2)
                axes[1].axhline(y=best_gd_loss, color='red', linestyle='--', label=f'GD (eta = {best_eta:.2f})')
                axes[1].set_xlabel('Step')
                axes[1].set_ylabel('MSE Loss')
                axes[1].set_title('Validation: TF vs GD')
                axes[1].legend()
                axes[1].grid(True, alpha=1)
                
                plt.tight_layout()
                plt.show()

    return model, losses, tf_val_losses, gd_val_losses, best_eta
