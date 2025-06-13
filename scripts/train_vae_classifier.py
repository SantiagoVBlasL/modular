#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_vae_classifier.py (Versión Tesis v6.0 - Diagnóstico y Metadatos)

Script definitivo para entrenar y evaluar modelos híbridos de β-VAE y clasificadores.
Esta versión introduce mejoras críticas para el diagnóstico y el rendimiento:
- Desglose de la pérdida VAE en Reconstrucción y KL.
- Gráficos automáticos del historial de entrenamiento del VAE por fold.
- Inclusión de metadatos (Edad, Sexo) como características para el clasificador.
- Logging detallado del balance de grupos y la composición de los vectores de características.
"""
from __future__ import annotations
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
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# --- Configuración del Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")

# --- 1. Definición del Modelo VAE Convolucional (ARQUITECTURA MODULAR v5.2) ---
class ConvolutionalVAE(nn.Module):
    def __init__(self,
                 input_channels: int,
                 latent_dim: int,
                 image_size: int,
                 num_conv_layers: int,
                 decoder_type: str,
                 kernel_sizes: List[int],
                 strides: List[int],
                 paddings: List[int],
                 conv_channels: List[int],
                 intermediate_fc_dim: int,
                 dropout_rate: float,
                 use_layernorm_fc: bool,
                 final_activation: str):
        super().__init__()
        self.latent_dim = latent_dim

        log.info("--- Construyendo Arquitectura VAE ---")
        log.info(f"Tamaño de entrada: ({input_channels}, {image_size}, {image_size})")

        # --- Encoder Dinámico ---
        encoder_layers = []
        current_ch = input_channels
        current_dim = image_size
        self.encoder_spatial_dims = [current_dim]

        log.info("-> Encoder Path:")
        for i in range(num_conv_layers):
            encoder_layers.extend([
                nn.Conv2d(current_ch, conv_channels[i], kernel_sizes[i], strides[i], paddings[i]),
                nn.ReLU(),
                nn.BatchNorm2d(conv_channels[i]),
                nn.Dropout2d(p=dropout_rate)
            ])
            prev_dim = current_dim
            current_ch = conv_channels[i]
            current_dim = ((current_dim + 2 * paddings[i] - kernel_sizes[i]) // strides[i]) + 1
            self.encoder_spatial_dims.append(current_dim)
            log.info(f"  Conv Layer {i+1}: ({prev_dim}x{prev_dim}) -> ({current_dim}x{current_dim}) | Canales: {conv_channels[i]}")

        self.encoder_conv = nn.Sequential(*encoder_layers)
        self.final_conv_channels = current_ch
        self.final_spatial_dim = current_dim
        self.flattened_size = self.final_conv_channels * self.final_spatial_dim**2
        log.info(f"  Tamaño Aplanado: {self.flattened_size}")

        fc_input_dim = self.flattened_size
        if intermediate_fc_dim > 0:
            self.encoder_fc = nn.Sequential(
                nn.Linear(self.flattened_size, intermediate_fc_dim),
                nn.LayerNorm(intermediate_fc_dim) if use_layernorm_fc else nn.Identity(),
                nn.ReLU(),
                nn.BatchNorm1d(intermediate_fc_dim),
                nn.Dropout(p=dropout_rate)
            )
            fc_input_dim = intermediate_fc_dim
            log.info(f"  Capa Intermedia FC: {self.flattened_size} -> {intermediate_fc_dim}")
        else:
            self.encoder_fc = nn.Identity()

        self.fc_mu = nn.Linear(fc_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(fc_input_dim, latent_dim)
        log.info(f"  Cuello de Botella (Latent Dim): {fc_input_dim} -> {latent_dim}")

        # --- Decoder Dinámico ---
        log.info("-> Decoder Path:")
        if intermediate_fc_dim > 0:
            self.decoder_fc = nn.Sequential(
                nn.Linear(latent_dim, intermediate_fc_dim),
                nn.LayerNorm(intermediate_fc_dim) if use_layernorm_fc else nn.Identity(),
                nn.ReLU(),
                nn.BatchNorm1d(intermediate_fc_dim),
                nn.Dropout(p=dropout_rate)
            )
            self.decoder_unflatten_fc = nn.Linear(intermediate_fc_dim, self.flattened_size)
            log.info(f"  Capa Intermedia FC: {latent_dim} -> {intermediate_fc_dim}")
            log.info(f"  Capa Unflatten FC: {intermediate_fc_dim} -> {self.flattened_size}")
        else:
            self.decoder_fc = nn.Identity()
            self.decoder_unflatten_fc = nn.Linear(latent_dim, self.flattened_size)
            log.info(f"  Capa Unflatten FC: {latent_dim} -> {self.flattened_size}")

        self.decoder_unflatten = nn.Unflatten(1, (self.final_conv_channels, self.final_spatial_dim, self.final_spatial_dim))
        
        decoder_layers = []
        current_ch = self.final_conv_channels
        reversed_channels = conv_channels[-2::-1] + [input_channels]
        
        reversed_kernels = kernel_sizes[::-1]
        reversed_strides = strides[::-1]
        reversed_paddings = paddings[::-1]

        if decoder_type == 'convtranspose':
            for i in range(num_conv_layers):
                in_dim = self.encoder_spatial_dims[num_conv_layers - i]
                out_dim = self.encoder_spatial_dims[num_conv_layers - 1 - i]
                k, s, p = reversed_kernels[i], reversed_strides[i], reversed_paddings[i]
                output_padding = out_dim - ((in_dim - 1) * s - 2 * p + k)
                
                decoder_layers.extend([
                    nn.ConvTranspose2d(current_ch, reversed_channels[i], k, s, p, output_padding=output_padding),
                    nn.ReLU() if i < num_conv_layers - 1 else nn.Identity(),
                    nn.BatchNorm2d(reversed_channels[i]) if i < num_conv_layers - 1 else nn.Identity(),
                    nn.Dropout2d(p=dropout_rate) if i < num_conv_layers - 1 else nn.Identity()
                ])
                log.info(f"  ConvTranspose Layer {i+1}: ({in_dim}x{in_dim}) -> ({out_dim}x{out_dim}) | Canales: {reversed_channels[i]}")
                current_ch = reversed_channels[i]
        else: # upsample_conv
            for i in range(num_conv_layers):
                in_dim = self.encoder_spatial_dims[num_conv_layers - i]
                out_dim = self.encoder_spatial_dims[num_conv_layers - 1 - i]
                decoder_layers.extend([
                    nn.Upsample(size=out_dim, mode='bilinear', align_corners=False),
                    nn.Conv2d(current_ch, reversed_channels[i], kernel_size=3, stride=1, padding=1),
                    nn.ReLU() if i < num_conv_layers - 1 else nn.Identity(),
                    nn.BatchNorm2d(reversed_channels[i]) if i < num_conv_layers - 1 else nn.Identity(),
                    nn.Dropout2d(p=dropout_rate) if i < num_conv_layers - 1 else nn.Identity()
                ])
                log.info(f"  Upsample+Conv Layer {i+1}: ({in_dim}x{in_dim}) -> ({out_dim}x{out_dim}) | Canales: {reversed_channels[i]}")
                current_ch = reversed_channels[i]

        if final_activation == 'tanh':
            decoder_layers.append(nn.Tanh())
        elif final_activation == 'sigmoid':
            decoder_layers.append(nn.Sigmoid())
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
        log.info(f"Tamaño de salida reconstruido: ({input_channels}, {image_size}, {image_size})")
        log.info("--- Fin Construcción Arquitectura VAE ---")

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        h = self.encoder_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = self.decoder_unflatten_fc(h)
        h = self.decoder_unflatten(h)
        return self.decoder_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        if recon_x.shape[-2:] != x.shape[-2:]:
             log.warning(f"Shape mismatch post-decoder! Input: {x.shape}, Recon: {recon_x.shape}. Interpolando a tamaño final.")
             recon_x = nn.functional.interpolate(recon_x, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return recon_x, mu, logvar, z

# --- 2. Funciones de Entrenamiento y Auxiliares ---

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
    total_loss = (recon_loss + beta * kld_loss) / x.size(0)
    return {
        'total': total_loss,
        'recon': recon_loss / x.size(0),
        'kld': (beta * kld_loss) / x.size(0)
    }

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
    log.info("Normalizando tensores con Z-score por canal (parámetros del train set):")
    for i in range(n_channels):
        mean, std = np.mean(train_tensors[:, i]), np.std(train_tensors[:, i])
        if std < 1e-6: std = 1.0
        train_tensors_norm[:, i] = (train_tensors[:, i] - mean) / std
        test_tensors_norm[:, i] = (test_tensors[:, i] - mean) / std
        log.info(f"  - Canal {i}: Media={mean:.3f}, Std={std:.3f}")
    return train_tensors_norm, test_tensors_norm

def plot_vae_training_history(history: dict, fold_idx: int, save_dir: Path):
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
    plt.savefig(save_dir / f"vae_training_history_fold_{fold_idx}.png", dpi=150)
    plt.close(fig)

def train_vae_for_fold(train_tensors, val_tensors, args, fold_idx_str, device):
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5, verbose=False) if val_loader and args.scheduler == 'plateau' else \
                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.lr_warmup_epochs) if args.scheduler == 'cosine' else None

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

            if scheduler:
                if args.scheduler == 'plateau': scheduler.step(avg_val_loss)
                elif epoch > args.lr_warmup_epochs: scheduler.step()

            if avg_val_loss < best_val_loss:
                best_val_loss, epochs_no_improve, best_model_state = avg_val_loss, 0, copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            if epoch % 20 == 0: 
                log.info(f"{fold_idx_str} | Epoch: {epoch:03d} | Train Loss: {history['train_loss'][-1]:.2f} (Rec:{history['train_recon'][-1]:.2f}|KLD:{history['train_kld'][-1]:.2f}) | Val Loss: {avg_val_loss:.2f} (Rec:{history['val_recon'][-1]:.2f}|KLD:{history['val_kld'][-1]:.2f}) | Beta: {current_beta:.3f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            if args.early_stopping > 0 and epochs_no_improve >= args.early_stopping:
                log.info(f"Early stopping en epoch {epoch}. Mejor Val Loss: {best_val_loss:.4f}"); break
    
    if best_model_state: model.load_state_dict(best_model_state)
    return model, history

def create_final_feature_vector(vae_model, tensors, scalar_features, age_sex_features, device):
    vae_model.eval()
    with torch.no_grad():
        mu, _ = vae_model.encode(torch.from_numpy(tensors).float().to(device))
    latent_vec = mu.cpu().numpy()
    
    log.info(f"Composición del vector de características final: Latente(VAE)={latent_vec.shape}, Escalares(Topo/HMM)={scalar_features.shape}, Metadatos(Edad/Sexo)={age_sex_features.shape}")
    return np.concatenate([latent_vec, scalar_features, age_sex_features], axis=1)

def get_classifier_and_grid(classifier_type, seed):
    if classifier_type == 'svm':
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

# --- 3. Función Principal de Orquestación ---

def main(args: argparse.Namespace):
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Configuración de la ejecución: {vars(args)}")
    
    dataset = load_full_dataset(Path(args.run_dir) / "data_for_cv", args.channels_to_use)
    if not dataset: return
    
    key_df = dataset['key_df']
    all_tensors = dataset['tensors']
    all_scalar_features = dataset['features']
    
    key_df['strat_key'] = key_df.apply(lambda row: '_'.join([str(row[col]) for col in args.stratify_on]), axis=1)
    y_labels = key_df['ResearchGroup'].astype('category').cat.codes.values
    
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    all_final_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(key_df)), key_df['strat_key'])):
        fold_idx_str = f"Fold {fold_idx + 1}/{args.n_folds}"
        log.info(f"--- Iniciando {fold_idx_str} ---")
        
        # --- División de datos y logging de balance ---
        df_train, df_test = key_df.iloc[train_idx], key_df.iloc[test_idx]
        log.info(f"Balance de grupos en {fold_idx_str} - TRAIN: \n{df_train[args.stratify_on].value_counts(normalize=True).sort_index()}")
        log.info(f"Balance de grupos en {fold_idx_str} - TEST: \n{df_test[args.stratify_on].value_counts(normalize=True).sort_index()}")

        X_train_tensors, X_test_tensors = all_tensors[train_idx], all_tensors[test_idx]
        X_train_scalar, X_test_scalar = all_scalar_features[train_idx], all_scalar_features[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]
        
        # Normalización de tensores
        X_train_tensors_norm, X_test_tensors_norm = normalize_tensors_in_fold(X_train_tensors, X_test_tensors)
        
        # Split para validación interna del VAE
        vae_train_tensors, vae_val_tensors, _, _ = train_test_split(
            X_train_tensors_norm, y_train, test_size=0.15, stratify=y_train, random_state=args.seed)
        
        # --- Manejo de Metadatos (Edad y Sexo) ---
        age_scaler = StandardScaler().fit(df_train[['Age']])
        X_train_age = age_scaler.transform(df_train[['Age']])
        X_test_age = age_scaler.transform(df_test[['Age']])
        
        sex_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(df_train[['Sex']])
        X_train_sex = sex_encoder.transform(df_train[['Sex']])
        X_test_sex = sex_encoder.transform(df_test[['Sex']])
        
        X_train_metadata = np.concatenate([X_train_age, X_train_sex], axis=1)
        X_test_metadata = np.concatenate([X_test_age, X_test_sex], axis=1)

        # Normalización de características escalares (Topo/HMM)
        scalar_scaler = StandardScaler().fit(X_train_scalar)
        X_train_scalar_scaled = scalar_scaler.transform(X_train_scalar)
        X_test_scalar_scaled = scalar_scaler.transform(X_test_scalar)
        
        # Entrenamiento del VAE
        vae_model, history = train_vae_for_fold(vae_train_tensors, vae_val_tensors, args, fold_idx_str, device)
        plot_vae_training_history(history, fold_idx + 1, output_dir)
        
        # Creación de vectores de características híbridos
        X_train_hybrid = create_final_feature_vector(vae_model, X_train_tensors_norm, X_train_scalar_scaled, X_train_metadata, device)
        X_test_hybrid = create_final_feature_vector(vae_model, X_test_tensors_norm, X_test_scalar_scaled, X_test_metadata, device)
        
        for clf_type in args.classifier_types:
            log.info(f"--- Entrenando y evaluando clasificador: {clf_type.upper()} ---")
            classifier, param_grid = get_classifier_and_grid(clf_type, args.seed)
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
            
            log.info(f"Ajustando hiperparámetros en el set de entrenamiento con {inner_cv.get_n_splits()}-Fold CV...")
            grid_search = GridSearchCV(classifier, param_grid, cv=inner_cv, scoring='balanced_accuracy', n_jobs=-1).fit(X_train_hybrid, y_train)
            log.info(f"Mejores parámetros para {clf_type}: {grid_search.best_params_}")
            
            final_clf = grid_search.best_estimator_
            y_pred_proba = final_clf.predict_proba(X_test_hybrid)
            
            metrics = {'fold': fold_idx + 1, 'classifier': clf_type, 'best_params': str(grid_search.best_params_)}
            try:
                metrics['auc_ovr'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except ValueError:
                metrics['auc_ovr'] = np.nan
            metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, final_clf.predict(X_test_hybrid))
            metrics['f1_macro'] = f1_score(y_test, final_clf.predict(X_test_hybrid), average='macro')
            all_final_metrics.append(metrics)
            log.info(f"Resultados {fold_idx_str} ({clf_type}): AUC_OVR={metrics.get('auc_ovr', 0):.4f}, Bal.Acc={metrics.get('balanced_accuracy', 0):.4f}")

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
        description="Entrenamiento de β-VAE + Clasificador Híbrido con CV y HP-Tuning (v6.0).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group_data = parser.add_argument_group('Data and Paths')
    group_data.add_argument('--run_dir', type=str, help="Ruta a la carpeta de la corrida que contiene 'data_for_cv'.")
    group_data.add_argument('--output_dir', type=str, help="Directorio para guardar resultados.")
    group_data.add_argument('--channels_to_use', type=int, nargs='*')
    
    group_vae_arch = parser.add_argument_group('VAE Architecture')
    group_vae_arch.add_argument('--latent_dim', type=int)
    group_vae_arch.add_argument('--num_conv_layers', type=int, choices=[3, 4])
    group_vae_arch.add_argument('--decoder_type', type=str, choices=['convtranspose', 'upsample_conv'])
    group_vae_arch.add_argument('--vae_conv_channels', type=int, nargs='+')
    group_vae_arch.add_argument('--vae_kernel_sizes', type=int, nargs='+')
    group_vae_arch.add_argument('--vae_paddings', type=int, nargs='+')
    group_vae_arch.add_argument('--vae_strides', type=int, nargs='+')
    group_vae_arch.add_argument('--intermediate_fc_dim', type=int)
    group_vae_arch.add_argument('--use_layernorm_fc', action='store_true')
    group_vae_arch.add_argument('--final_activation', type=str, choices=['tanh', 'sigmoid', 'linear'])
    
    group_train = parser.add_argument_group('Training Parameters')
    group_train.add_argument('--beta', type=float)
    group_train.add_argument('--dropout_rate', type=float)
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
    group_cv.add_argument('--classifier_types', type=str, nargs='+', choices=['rf', 'svm', 'logreg', 'gb', 'mlp'])
    group_cv.add_argument('--stratify_on', type=str, nargs='+')
    
    group_general = parser.add_argument_group('General')
    group_general.add_argument('--seed', type=int)
    
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)
    
    if not args.run_dir:
        parser.error("El argumento --run_dir es obligatorio (o debe estar en el archivo de configuración).")

    main(args)