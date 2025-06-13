# training/models.py
"""
Define las arquitecturas de redes neuronales utilizadas en el proyecto,
comenzando con el Autoencoder Variacional Convolucional (VAE).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import logging
from typing import List

log = logging.getLogger(__name__)

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