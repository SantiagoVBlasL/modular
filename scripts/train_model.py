#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model.py (Versión Tesis v7.0 - Modular)

Script principal para orquestar el entrenamiento y evaluación de modelos
híbridos de VAE y clasificadores. Este script utiliza el paquete `training`
para mantener el código organizado y modular.
"""
from __future__ import annotations  # Para compatibilidad con Python 3.7+
import sys
import yaml
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- Añadir el directorio raíz del proyecto al path ---
# Esto permite importar los módulos del paquete 'training'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from training import data_handling, models, trainer, utils

# --- Configuración del Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

def main(args: argparse.Namespace):
    """
    Función principal que orquesta todo el pipeline de entrenamiento y evaluación.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Dispositivo seleccionado: {device}")
    log.info(f"Configuración de la ejecución: {vars(args)}")

    # --- 1. Carga y Pre-procesado de Datos ---
    dataset = data_handling.load_full_dataset(Path(args.run_dir) / "data_for_cv", args.channels_to_use)
    if not dataset:
        log.critical("No se pudieron cargar los datos. Abortando.")
        return
    
    dataset['tensors'] = data_handling.preprocess_tensors_robustly(dataset['tensors'])
    
    key_df = dataset['key_df']
    all_tensors = dataset['tensors']
    all_scalar_features = dataset['features']
    
    key_df['strat_key'] = key_df.apply(lambda row: '_'.join([str(row[col]) for col in args.stratify_on]), axis=1)
    y_labels = key_df['ResearchGroup'].astype('category').cat.codes.values
    
    # --- 2. Bucle de Validación Cruzada ---
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    all_final_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(key_df)), key_df['strat_key'])):
        fold_idx_str = f"Fold {fold_idx + 1}/{args.n_folds}"
        log.info(f"--- Iniciando {fold_idx_str} ---")

        # División de datos y logging de balance
        df_train, df_test = key_df.iloc[train_idx], key_df.iloc[test_idx]
        log.info(f"Balance de grupos en {fold_idx_str} - TRAIN:\n{df_train[args.stratify_on].value_counts(normalize=True).sort_index()}")
        log.info(f"Balance de grupos en {fold_idx_str} - TEST:\n{df_test[args.stratify_on].value_counts(normalize=True).sort_index()}")

        X_train_tensors, X_test_tensors = all_tensors[train_idx], all_tensors[test_idx]
        X_train_scalar, X_test_scalar = all_scalar_features[train_idx], all_scalar_features[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]

        # Split para validación interna del VAE
        vae_train_tensors, vae_val_tensors, _, _ = train_test_split(
            X_train_tensors, y_train, test_size=0.15, stratify=y_train, random_state=args.seed)

        # Manejo de Metadatos (Edad y Sexo)
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

        # --- 3. Entrenamiento del VAE ---
        vae_model, history = trainer.train_vae_for_fold(vae_train_tensors, vae_val_tensors, args, fold_idx_str, device)
        utils.plot_vae_training_history(history, fold_idx + 1, output_dir)

        # --- 4. Creación de Vectores Híbridos ---
        X_train_hybrid = data_handling.create_final_feature_vector(vae_model, X_train_tensors, X_train_scalar_scaled, X_train_metadata, device)
        X_test_hybrid = data_handling.create_final_feature_vector(vae_model, X_test_tensors, X_test_scalar_scaled, X_test_metadata, device)

        # --- 5. Entrenamiento de Clasificadores ---
        fold_metrics = trainer.train_and_evaluate_classifiers(
            X_train_hybrid, y_train, X_test_hybrid, y_test,
            args.classifier_types, args.seed, fold_idx_str
        )
        all_final_metrics.extend(fold_metrics)

    # --- 6. Reporte Final ---
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
        description="Entrenamiento de β-VAE + Clasificador Híbrido con CV y HP-Tuning (v7.0).",
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
    group_cv.add_argument('--classifier_types', type=str, nargs='+', choices=['rf', 'svm', 'logreg', 'gb', 'mlp', 'lgbm'])
    group_cv.add_argument('--stratify_on', type=str, nargs='+')
    
    group_general = parser.add_argument_group('General')
    group_general.add_argument('--seed', type=int)
    
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)
    
    if not args.run_dir:
        parser.error("El argumento --run_dir es obligatorio (o debe estar en el archivo de configuración).")

    main(args)