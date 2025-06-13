# training/utils.py
"""
Funciones de utilidad para el pipeline de entrenamiento, incluyendo la pérdida,
schedulers, gráficos y la definición de los clasificadores y sus grillas de búsqueda.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

def get_cyclical_beta_schedule(current_epoch, total_epochs, beta_max, n_cycles, start_epoch):
    """Calcula el valor de beta para el epoch actual usando un schedule cíclico."""
    if current_epoch < start_epoch: return 0.0
    effective_epoch = current_epoch - start_epoch
    effective_total = total_epochs - start_epoch
    if n_cycles <= 0: return beta_max
    epoch_per_cycle = max(1, effective_total / n_cycles)
    epoch_in_cycle = effective_epoch % epoch_per_cycle
    return beta_max * (epoch_in_cycle / epoch_per_cycle)

def vae_loss_function(recon_x, x, mu, logvar, beta):
    """Calcula la pérdida del VAE (Reconstrucción + KLD) y la desglosa."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = (recon_loss + beta * kld_loss) / x.size(0)
    return {
        'total': total_loss,
        'recon': recon_loss / x.size(0),
        'kld': (beta * kld_loss) / x.size(0)
    }

def plot_vae_training_history(history: dict, fold_idx: int, save_dir: Path):
    """Genera y guarda un gráfico del historial de entrenamiento del VAE."""
    fig, ax1 = plt.subplots(figsize=(12, 7))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Eje de Pérdidas
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss (Total)')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss (Total)')
    ax1.plot(epochs, history['train_recon'], 'b--', alpha=0.6, label='Train Loss (Recon)')
    ax1.plot(epochs, history['val_recon'], 'r--', alpha=0.6, label='Validation Loss (Recon)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Pérdida', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Eje de KLD
    ax2 = ax1.twinx()
    ax2.plot(epochs, history['train_kld'], 'g:', label='Train Loss (KLD)')
    ax2.plot(epochs, history['val_kld'], 'm:', label='Validation Loss (KLD)')
    ax2.set_ylabel('Pérdida KLD', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.legend(loc='upper right')

    plt.title(f'Historial de Entrenamiento VAE - Fold {fold_idx}', fontsize=16)
    plt.savefig(save_dir / f"vae_training_history_fold_{fold_idx}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

def get_classifier_and_grid(classifier_type: str, seed: int):
    """Devuelve una instancia del clasificador y su grilla de hiperparámetros para GridSearchCV."""
    if classifier_type == 'lgbm':
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM no está instalado. Por favor, ejecute 'pip install lightgbm'.")
        return LGBMClassifier(random_state=seed, objective='multiclass', class_weight='balanced'), {
            'n_estimators': [100, 200, 400], 'learning_rate': [0.01, 0.05, 0.1], 'num_leaves': [21, 31, 41]
        }
    elif classifier_type == 'svm':
        return SVC(probability=True, random_state=seed, class_weight='balanced'), {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 'scale'], 'kernel': ['rbf']}
    elif classifier_type == 'rf':
        return RandomForestClassifier(random_state=seed, class_weight='balanced', n_jobs=-1), {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}
    elif classifier_type == 'logreg':
        return LogisticRegression(random_state=seed, class_weight='balanced', solver='liblinear', max_iter=1000), {'C': [0.001, 0.01, 0.1, 1, 10]}
    elif classifier_type == 'gb':
        return GradientBoostingClassifier(random_state=seed), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    elif classifier_type == 'mlp':
         return MLPClassifier(random_state=seed, max_iter=750, early_stopping=True, n_iter_no_change=20), {'hidden_layer_sizes': [(128,64), (100,), (50, 25)], 'alpha': [0.0001, 0.001, 0.01], 'learning_rate_init': [0.001, 0.005]}
    raise ValueError(f"Clasificador no soportado: {classifier_type}")