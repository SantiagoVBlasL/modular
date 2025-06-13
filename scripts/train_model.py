#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model.py (v8.0 - Robusto y Reproducible)

Script principal para orquestar el entrenamiento y evaluación de modelos
híbridos de VAE y clasificadores.

Esta versión está diseñada para consumir los artefactos pre-procesados
generados por 'prepare_and_analyze_data.py', garantizando un pipeline
robusto y 100% reproducible.

Funcionalidades Clave:
- Carga artefactos de datos autocontenidos (tensores, scalers, metadata).
- Valida la integridad de los datos mediante hash SHA-256.
- Sincroniza la configuración del experimento con los metadatos del artefacto.
- Elimina cualquier reprocesamiento de datos en tiempo de ejecución.
"""
from __future__ import annotations
import sys
import yaml
import logging
import argparse
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from joblib import load


project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from training import models, trainer, utils
from training import data_handling

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

def load_and_validate_artifacts(data_dir: Path, args: argparse.Namespace) -> dict | None:
    """
    Carga todos los artefactos de datos y metadatos desde el directorio
    pre-procesado, validando su integridad.
    """
    log.info(f"Cargando artefactos pre-procesados desde: {data_dir}")
    
    try:
        with open(data_dir / "meta.yaml", 'r') as f:
            meta = yaml.safe_load(f)

        if args.final_activation and args.final_activation != meta.get("recommended_final_activation"):
            log.warning(
                f"Conflicto de activación: El experimento pide '{args.final_activation}' pero los datos "
                f"fueron preparados para '{meta['recommended_final_activation']}'. Se usará el valor del experimento."
            )
        else:
            args.final_activation = meta.get("recommended_final_activation", "tanh")
        log.info(f"Activación final confirmada para el VAE: {args.final_activation}")

        all_tensors_preprocessed = np.load(data_dir / "tensors_preprocessed.npy")
        
        tensor_hash_check = hashlib.sha256(all_tensors_preprocessed.tobytes()).hexdigest()
        assert tensor_hash_check == meta['tensor_sha256'], "¡El hash de los tensores no coincide! Los datos pueden estar corruptos."
        log.info("Verificación de integridad de tensores (SHA-256) superada.")

        dataset = {
            'key_df': pd.read_csv(data_dir / "cv_subjects_key.csv"),
            'tensors': all_tensors_preprocessed,
            'features_unscaled': np.load(data_dir / "features_unscaled.npy"),
            'scalar_scaler': load(data_dir / "scalar_features_scaler.pkl"),
            'age_scaler': load(data_dir / "age_scaler.pkl"),
            'sex_encoder': load(data_dir / "sex_encoder.pkl"),
            'meta': meta
        }
        
        # Selección de canales si se especifica
        channels_to_use = args.channels_to_use
        if channels_to_use:
            log.info(f"Seleccionando canales en los índices: {channels_to_use}")
            dataset['tensors'] = dataset['tensors'][:, channels_to_use, :, :]

        return dataset

    except FileNotFoundError as e:
        log.critical(f"Error: No se encontró el artefacto esperado en '{data_dir}'. Detalle: {e}")
        log.critical("Asegúrate de haber ejecutado 'prepare_and_analyze_data.py' primero.")
        return None
    except AssertionError as e:
        log.critical(f"Error de validación: {e}")
        return None


def main(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Dispositivo seleccionado: {device}")
    log.info(f"Configuración de la ejecución: {vars(args)}")

    data_dir = Path(args.run_dir) / "data_for_cv"
    dataset = load_and_validate_artifacts(data_dir, args)
    if not dataset:
        return

    key_df = dataset['key_df']
    all_tensors = dataset['tensors']
    all_scalar_features_unscaled = dataset['features_unscaled']
    
    key_df['strat_key'] = key_df.apply(lambda row: '_'.join([str(row[col]) for col in args.stratify_on]), axis=1)
    y_labels = key_df['ResearchGroup'].astype('category').cat.codes.values
    
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    all_final_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(key_df)), key_df['strat_key'])):
        fold_idx_str = f"Fold {fold_idx + 1}/{args.n_folds}"
        log.info(f"--- Iniciando {fold_idx_str} ---")

        df_train, df_test = key_df.iloc[train_idx], key_df.iloc[test_idx]
        X_train_tensors, X_test_tensors = all_tensors[train_idx], all_tensors[test_idx]
        X_train_scalar_unscaled, X_test_scalar_unscaled = all_scalar_features_unscaled[train_idx], all_scalar_features_unscaled[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]

        assert X_train_tensors.min() >= -1.0 and X_train_tensors.max() <= 1.0, f"Fold {fold_idx}: Los tensores de entrenamiento están fuera del rango [-1, 1]"

        vae_train_tensors, vae_val_tensors, _, _ = train_test_split(
            X_train_tensors, y_train, test_size=0.15, stratify=y_train, random_state=args.seed)

        X_train_age = dataset['age_scaler'].transform(df_train[['Age']])
        X_test_age = dataset['age_scaler'].transform(df_test[['Age']])
        
        X_train_sex = dataset['sex_encoder'].transform(df_train[['Sex']])
        X_test_sex = dataset['sex_encoder'].transform(df_test[['Sex']])
        
        X_train_metadata = np.concatenate([X_train_age, X_train_sex], axis=1)
        X_test_metadata = np.concatenate([X_test_age, X_test_sex], axis=1)

        X_train_scalar_scaled = dataset['scalar_scaler'].transform(X_train_scalar_unscaled)
        X_test_scalar_scaled = dataset['scalar_scaler'].transform(X_test_scalar_unscaled)

        vae_model, history = trainer.train_vae_for_fold(vae_train_tensors, vae_val_tensors, args, fold_idx_str, device)
        utils.plot_vae_training_history(history, fold_idx + 1, output_dir)

        X_train_hybrid = data_handling.create_final_feature_vector(vae_model, X_train_tensors, X_train_scalar_scaled, X_train_metadata, device)
        X_test_hybrid = data_handling.create_final_feature_vector(vae_model, X_test_tensors, X_test_scalar_scaled, X_test_metadata, device)

        fold_metrics = trainer.train_and_evaluate_classifiers(
            X_train_hybrid, y_train, X_test_hybrid, y_test,
            args.classifier_types, args.seed, fold_idx_str
        )
        all_final_metrics.extend(fold_metrics)

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
        description="Entrenamiento de β-VAE + Clasificador Híbrido con CV y HP-Tuning (v8.0).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group_data = parser.add_argument_group('Data and Paths')
    group_data.add_argument('--run_dir', type=str, help="Ruta a la carpeta de la corrida que contiene el directorio 'data_for_cv'.")
    group_data.add_argument('--output_dir', type=str, help="Directorio para guardar resultados del entrenamiento.")
    group_data.add_argument('--channels_to_use', type=int, nargs='*', help="Índices de canales a seleccionar del tensor pre-procesado.")
    
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
    group_vae_arch.add_argument('--final_activation', type=str, choices=['tanh', 'sigmoid', 'linear'], help="Si no se especifica, se usará el valor de meta.yaml.")
    
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
        parser.error("--run_dir es obligatorio, ya sea como argumento de línea de comandos o en el archivo de configuración.")

    main(args)
