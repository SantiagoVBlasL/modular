# training/trainer.py
"""
Lógica central de entrenamiento y evaluación para el pipeline VAE-Clasificador.
"""
from __future__ import annotations
import torch.nn as nn
import logging
import torch
import torch.optim as optim
import copy
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

# Importaciones desde otros módulos del paquete
from .models import ConvolutionalVAE
from .utils import get_cyclical_beta_schedule, vae_loss_function, get_classifier_and_grid

log = logging.getLogger(__name__)

def train_vae_for_fold(train_tensors, val_tensors, args, fold_idx_str, device):
    """Entrena y valida un modelo VAE para un único fold de la validación cruzada."""
    log.info(f"Iniciando entrenamiento del β-VAE en {len(train_tensors)} muestras, validando con {len(val_tensors)}.")
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_tensors).float()), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_tensors).float()), batch_size=args.batch_size) if val_tensors is not None and len(val_tensors) > 0 else None
    
    model = ConvolutionalVAE(
        input_channels=train_tensors.shape[1], image_size=train_tensors.shape[2], latent_dim=args.latent_dim,
        num_conv_layers=args.num_conv_layers, decoder_type=args.decoder_type, kernel_sizes=args.vae_kernel_sizes,
        strides=args.vae_strides, paddings=args.vae_paddings, conv_channels=args.vae_conv_channels,
        intermediate_fc_dim=args.intermediate_fc_dim, dropout_rate=args.dropout_rate,
        use_layernorm_fc=args.use_layernorm_fc, final_activation=args.final_activation
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) if args.optimizer == 'adamw' else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.lr_warmup_epochs) if args.scheduler == 'cosine' else None

    best_val_loss, epochs_no_improve, best_model_state = float('inf'), 0, None
    history = {'train_loss': [], 'val_loss': [], 'train_recon': [], 'val_recon': [], 'train_kld': [], 'val_kld': []}

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.lr_warmup_epochs:
            warmup_factor = epoch / max(1, args.lr_warmup_epochs)
            for g in optimizer.param_groups: g['lr'] = args.lr * warmup_factor
        
        model.train()
        current_beta = get_cyclical_beta_schedule(epoch, args.epochs, args.beta, args.beta_cycles, args.kl_start_epoch)
        train_loss, train_recon, train_kld = 0, 0, 0
        for data_batch, in train_loader:
            data_batch = data_batch.to(device); optimizer.zero_grad()
            recon_batch, mu, logvar, _ = model(data_batch)
            loss_dict = vae_loss_function(recon_batch, data_batch, mu, logvar, current_beta)
            loss = loss_dict['total']
            loss.backward()
            if args.clip_grad_norm: nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            train_loss += loss.item() * data_batch.size(0)
            train_recon += loss_dict['recon'].item() * data_batch.size(0)
            train_kld += loss_dict['kld'].item() * data_batch.size(0)

        history['train_loss'].append(train_loss / len(train_loader.dataset))
        history['train_recon'].append(train_recon / len(train_loader.dataset))
        history['train_kld'].append(train_kld / len(train_loader.dataset))
        
        val_loss, val_recon, val_kld = 0, 0, 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for val_data, in val_loader:
                    val_data = val_data.to(device)
                    recon, mu, logvar, _ = model(val_data)
                    loss_dict = vae_loss_function(recon, val_data, mu, logvar, current_beta)
                    val_loss += loss_dict['total'].item() * val_data.size(0)
                    val_recon += loss_dict['recon'].item() * val_data.size(0)
                    val_kld += loss_dict['kld'].item() * val_data.size(0)
            avg_val_loss = val_loss / len(val_loader.dataset)
            history['val_loss'].append(avg_val_loss)
            history['val_recon'].append(val_recon / len(val_loader.dataset))
            history['val_kld'].append(val_kld / len(val_loader.dataset))

            if scheduler and epoch > args.lr_warmup_epochs: scheduler.step()

            if avg_val_loss < best_val_loss:
                best_val_loss, epochs_no_improve, best_model_state = avg_val_loss, 0, copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            if epoch % 20 == 0: 
                log.info(f"{fold_idx_str} | Epoch: {epoch:03d} | Train Loss: {history['train_loss'][-1]:.4f} (Rec:{history['train_recon'][-1]:.4f}|KLD:{history['train_kld'][-1]:.4f}) | Val Loss: {avg_val_loss:.4f} (Rec:{history['val_recon'][-1]:.4f}|KLD:{history['val_kld'][-1]:.4f}) | Beta: {current_beta:.3f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            if args.early_stopping > 0 and epochs_no_improve >= args.early_stopping:
                log.info(f"Early stopping en epoch {epoch}. Mejor Val Loss: {best_val_loss:.4f}"); break
    
    if best_model_state: model.load_state_dict(best_model_state)
    return model, history

def train_and_evaluate_classifiers(X_train, y_train, X_test, y_test, classifier_types, seed, fold_idx_str):
    """
    Entrena y evalúa múltiples tipos de clasificadores usando GridSearchCV.
    """
    fold_metrics = []
    for clf_type in classifier_types:
        log.info(f"--- Entrenando y evaluando clasificador: {clf_type.upper()} ---")
        classifier, param_grid = get_classifier_and_grid(clf_type, seed)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        
        log.info(f"Ajustando hiperparámetros en el set de entrenamiento con {inner_cv.get_n_splits()}-Fold CV...")
        grid_search = GridSearchCV(classifier, param_grid, cv=inner_cv, scoring='balanced_accuracy', n_jobs=-1, verbose=1).fit(X_train, y_train)
        log.info(f"Mejores parámetros para {clf_type}: {grid_search.best_params_}")
        
        final_clf = grid_search.best_estimator_
        y_pred_proba = final_clf.predict_proba(X_test)
        
        metrics = {'fold': int(fold_idx_str.split(' ')[1].split('/')[0]), 'classifier': clf_type, 'best_params': str(grid_search.best_params_)}
        try:
            metrics['auc_ovr'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except ValueError:
            metrics['auc_ovr'] = np.nan
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, final_clf.predict(X_test))
        metrics['f1_macro'] = f1_score(y_test, final_clf.predict(X_test), average='macro')
        fold_metrics.append(metrics)
        log.info(f"Resultados {fold_idx_str} ({clf_type}): AUC_OVR={metrics.get('auc_ovr', 0):.4f}, Bal.Acc={metrics.get('balanced_accuracy', 0):.4f}")
    return fold_metrics