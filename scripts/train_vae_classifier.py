#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_vae_classifier.py (Versión Tesis v4.0 - Profesional)

Script definitivo para entrenar y evaluar modelos híbridos de β-VAE y clasificadores,
utilizando una validación cruzada anidada y estratificada.
Este script soporta archivos de configuración YAML para una gestión de experimentos
profesional, reproducible y flexible, con control total sobre el entrenamiento.
"""
from __future__ import annotations  # Para compatibilidad con tipos de retorno en Python 3.7+
import argparse
import gc
import logging
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import yaml
from typing import List, Dict, Tuple

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

from torch.utils.data import DataLoader, TensorDataset

# --- Configuración del Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

# --- 1. Definición del Modelo VAE Convolucional (Versión Flexible) ---
class ConvolutionalVAE(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int, image_size: int, dropout_rate: float, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1), nn.ReLU(), nn.BatchNorm2d(32), nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), nn.BatchNorm2d(64), nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), nn.BatchNorm2d(128), nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Flatten()
        )
        with torch.no_grad():
            self.flattened_size = self.encoder_conv(torch.zeros(1, input_channels, image_size, image_size)).shape[1]
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, self.flattened_size)
        self.decoder_unflatten = nn.Unflatten(1, (256, 8, 8))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 5, 2, 2, output_padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, input_channels, 5, 2, 2, output_padding=1), nn.Tanh()
        )
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar); eps = torch.randn_like(std); return mu + eps * std
    def encode(self, x):
        h = self.encoder_conv(x); return self.fc_mu(h), self.fc_logvar(h)
    def decode(self, z):
        h = self.decoder_fc(z); h = self.decoder_unflatten(h); return self.decoder_conv(h)
    def forward(self, x):
        mu, logvar = self.encode(x); z = self.reparameterize(mu, logvar); recon_x = self.decode(z)
        if recon_x.shape != x.shape:
            recon_x = nn.functional.interpolate(recon_x, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return recon_x, mu, logvar, z

def get_cyclical_beta_schedule(current_epoch, total_epochs, beta_max, n_cycles, start_epoch):
    if current_epoch < start_epoch: return 0.0
    effective_epoch = current_epoch - start_epoch
    effective_total = total_epochs - start_epoch
    if n_cycles <= 0: return beta_max
    epoch_per_cycle = max(1, effective_total / n_cycles)
    epoch_in_cycle = effective_epoch % epoch_per_cycle
    return beta_max * (epoch_in_cycle / epoch_per_cycle)

def vae_loss_function(recon_x, x, mu, logvar, beta):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kld_loss) / x.size(0)

# --- 2. Lógica de Entrenamiento y Evaluación ---

def load_full_dataset(data_dir: Path, channel_indices: list[int] | None) -> dict | None:
    log.info(f"Cargando dataset completo para CV desde: {data_dir}")
    try:
        data = {
            'key_df': pd.read_csv(data_dir / "cv_subjects_key.csv"),
            'tensors': np.load(data_dir / "cv_all_tensors.npy"),
            'features': np.load(data_dir / "cv_all_features_unscaled.npy"),
        }
        if channel_indices:
            log.info(f"Seleccionando canales en los índices: {channel_indices}"); data['tensors'] = data['tensors'][:, channel_indices, :, :]
        return data
    except FileNotFoundError as e:
        log.error(f"Error: Archivo no encontrado. Ejecuta 'prepare_and_analyze_data.py' primero. Detalle: {e}")
        return None

def normalize_tensors_in_fold(train_tensors, test_tensors):
    n_channels = train_tensors.shape[1]
    train_tensors_norm, test_tensors_norm = np.zeros_like(train_tensors), np.zeros_like(test_tensors)
    for i in range(n_channels):
        mean, std = np.mean(train_tensors[:, i]), np.std(train_tensors[:, i])
        if std < 1e-6: std = 1.0
        train_tensors_norm[:, i] = (train_tensors[:, i] - mean) / std; test_tensors_norm[:, i] = (test_tensors[:, i] - mean) / std
    log.info("Tensores de entrada normalizados (z-score por canal) dentro del fold.")
    return train_tensors_norm, test_tensors_norm

def train_vae_for_fold(train_tensors, val_tensors, args, fold_idx_str, device):
    log.info(f"Iniciando entrenamiento del β-VAE en {len(train_tensors)} muestras, validando con {len(val_tensors)}.")
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_tensors).float()), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_tensors).float()), batch_size=args.batch_size) if len(val_tensors) > 0 else None
    
    model = ConvolutionalVAE(input_channels=train_tensors.shape[1], latent_dim=args.latent_dim, image_size=train_tensors.shape[2], dropout_rate=args.dropout_rate).to(device)
    
    if args.optimizer == 'adamw': optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else: optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    if val_loader and args.scheduler == 'plateau': scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5, verbose=False)
    elif args.scheduler == 'cosine': scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.lr_warmup_epochs)
    else: scheduler = None

    best_val_loss, epochs_no_improve, best_model_state = float('inf'), 0, None
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.lr_warmup_epochs:
            warmup_factor = epoch / max(1, args.lr_warmup_epochs)
            for g in optimizer.param_groups: g['lr'] = args.lr * warmup_factor
        
        model.train()
        current_beta = get_cyclical_beta_schedule(epoch, args.epochs, args.beta, args.beta_cycles, args.kl_start_epoch)
        for data_batch, in train_loader:
            data_batch = data_batch.to(device); optimizer.zero_grad()
            recon_batch, mu, logvar, _ = model(data_batch)
            loss = vae_loss_function(recon_batch, data_batch, mu, logvar, current_beta)
            loss.backward()
            if args.clip_grad_norm: nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
        
        if val_loader:
            model.eval(); val_loss = 0
            with torch.no_grad():
                for val_data, in val_loader:
                    recon, mu, logvar, _ = model(val_data.to(device))
                    val_loss += vae_loss_function(recon, val_data.to(device), mu, logvar, current_beta).item() * val_data.size(0)
            avg_val_loss = val_loss / len(val_loader.dataset)
            if scheduler and epoch > args.lr_warmup_epochs: scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss, epochs_no_improve, best_model_state = avg_val_loss, 0, copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            if epoch % 20 == 0: log.info(f"{fold_idx_str} | Epoch: {epoch} | Val Loss: {avg_val_loss:.4f} | Beta: {current_beta:.3f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            if args.early_stopping > 0 and epochs_no_improve >= args.early_stopping:
                log.info(f"Early stopping en epoch {epoch}. Mejor Val Loss: {best_val_loss:.4f}"); break
    
    if best_model_state: model.load_state_dict(best_model_state)
    return model

def create_hybrid_features(vae_model, tensors, scalar_features, device):
    vae_model.eval()
    with torch.no_grad():
        mu, _ = vae_model.encode(torch.from_numpy(tensors).float().to(device))
    return np.concatenate([mu.cpu().numpy(), scalar_features], axis=1)

def get_classifier_and_grid(classifier_type, seed):
    if classifier_type == 'svm':
        return SVC(probability=True, random_state=seed, class_weight='balanced'), \
               {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 'scale'], 'kernel': ['rbf']}
    elif classifier_type == 'rf':
        return RandomForestClassifier(random_state=seed, class_weight='balanced', n_jobs=-1), \
               {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}
    elif classifier_type == 'logreg':
        return LogisticRegression(random_state=seed, class_weight='balanced', solver='liblinear', max_iter=1000), \
               {'C': [0.001, 0.01, 0.1, 1, 10]}
    raise ValueError(f"Clasificador no soportado: {classifier_type}")

def main(args: argparse.Namespace):
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Configuración de la ejecución: {vars(args)}")
    
    dataset = load_full_dataset(Path(args.run_dir) / "data_for_cv", args.channels_to_use)
    if not dataset: return

    key_df, all_tensors, all_features = dataset['key_df'], dataset['tensors'], dataset['features']
    key_df['strat_key'] = key_df[args.stratify_on].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    
    y_labels = key_df['ResearchGroup'].astype('category').cat.codes.values
    
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    all_final_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(key_df, key_df['strat_key'])):
        log.info(f"--- Iniciando Fold {fold_idx + 1}/{args.n_folds} ---")
        
        X_train_tensors, X_test_tensors = all_tensors[train_idx], all_tensors[test_idx]
        X_train_features, X_test_features = all_features[train_idx], all_features[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]

        X_train_tensors_norm, X_test_tensors_norm = normalize_tensors_in_fold(X_train_tensors, X_test_tensors)
        
        vae_train_tensors, vae_val_tensors, _, _ = train_test_split(
            X_train_tensors_norm, y_train, test_size=0.15, stratify=y_train, random_state=args.seed)
        
        scaler = StandardScaler().fit(X_train_features)
        X_train_features_scaled, X_test_features_scaled = scaler.transform(X_train_features), scaler.transform(X_test_features)
        
        vae_model = train_vae_for_fold(vae_train_tensors, vae_val_tensors, args, f"Fold {fold_idx + 1}", device)
        
        X_train_hybrid = create_hybrid_features(vae_model, X_train_tensors_norm, X_train_features_scaled, device)
        X_test_hybrid = create_hybrid_features(vae_model, X_test_tensors_norm, X_test_features_scaled, device)
        
        for clf_type in args.classifier_types:
            log.info(f"--- Entrenando y evaluando clasificador: {clf_type.upper()} ---")
            classifier, param_grid = get_classifier_and_grid(clf_type, args.seed)
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
            grid_search = GridSearchCV(classifier, param_grid, cv=inner_cv, scoring='balanced_accuracy', n_jobs=-1).fit(X_train_hybrid, y_train)
            log.info(f"Mejores parámetros para {clf_type}: {grid_search.best_params_}")
            
            final_clf = grid_search.best_estimator_
            y_pred_proba = final_clf.predict_proba(X_test_hybrid)
            
            metrics = {'fold': fold_idx + 1, 'classifier': clf_type, 'best_params': str(grid_search.best_params_)}
            try: metrics['auc_ovr'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except ValueError: metrics['auc_ovr'] = np.nan
            metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, final_clf.predict(X_test_hybrid))
            metrics['f1_macro'] = f1_score(y_test, final_clf.predict(X_test_hybrid), average='macro')
            all_final_metrics.append(metrics)
            log.info(f"Resultados Fold {fold_idx + 1} ({clf_type}): AUC_OVR={metrics.get('auc_ovr', 0):.4f}, Bal.Acc={metrics.get('balanced_accuracy', 0):.4f}")

    metrics_df = pd.DataFrame(all_final_metrics)
    log.info("\n--- Resumen Final de Rendimiento (Promedio sobre Folds) ---")
    log.info(f"\n{metrics_df.groupby('classifier').mean(numeric_only=True).to_string()}")
    metrics_df.to_csv(output_dir / "final_performance_metrics.csv", index=False)
    log.info(f"\nResultados detallados guardados en: {output_dir / 'final_performance_metrics.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, help='Ruta a un archivo de configuración YAML.')
    config_args, remaining_argv = parser.parse_known_args()
    
    defaults = {}
    if config_args.config:
        with open(config_args.config, 'r') as f:
            defaults = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        parents=[parser],
        description="Entrenamiento de β-VAE + Clasificador Híbrido con CV y HP-Tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group_data = parser.add_argument_group('Data and Paths')
    group_data.add_argument('--run_dir', type=str, help="Ruta a la carpeta de la corrida que contiene 'data_for_cv'.")
    group_data.add_argument('--output_dir', type=str, help="Directorio para guardar resultados.")
    group_data.add_argument('--channels_to_use', type=int, nargs='*')
    
    group_vae = parser.add_argument_group('VAE Hyperparameters')
    group_vae.add_argument('--latent_dim', type=int)
    group_vae.add_argument('--beta', type=float)
    group_vae.add_argument('--dropout_rate', type=float)
    
    group_train = parser.add_argument_group('Training Parameters')
    group_train.add_argument('--epochs', type=int)
    group_train.add_argument('--lr', type=float)
    group_train.add_argument('--batch_size', type=int)
    group_train.add_argument('--weight_decay', type=float)
    group_train.add_argument('--early_stopping', type=int)
    group_train.add_argument('--beta_cycles', type=int)
    group_train.add_argument('--kl_start_epoch', type=int)
    group_train.add_argument('--optimizer', type=str, choices=['adam', 'adamw'])
    group_train.add_argument('--scheduler', type=str, choices=['plateau', 'cosine', 'none'])
    group_train.add_argument('--lr_warmup_epochs', type=int)
    group_train.add_argument('--clip_grad_norm', type=float)
    
    group_cv = parser.add_argument_group('Cross-Validation & Classifier')
    group_cv.add_argument('--n_folds', type=int)
    group_cv.add_argument('--classifier_types', type=str, nargs='+', choices=['rf', 'svm', 'logreg'])
    group_cv.add_argument('--stratify_on', type=str, nargs='+')
    
    group_general = parser.add_argument_group('General')
    group_general.add_argument('--seed', type=int)
    
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)
    
    if not args.run_dir:
        parser.error("El argumento --run_dir es obligatorio (o debe estar en el archivo de configuración).")

    main(args)