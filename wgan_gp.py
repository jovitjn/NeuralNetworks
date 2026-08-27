"""Model and loss components for a 64x64 Wasserstein GAN with gradient penalty."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Critic(nn.Module):
    """Convolutional critic that maps 64x64 images to unconstrained scores."""

    def __init__(self, image_channels: int = 1, features: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(image_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(features, features * 2),
            self._block(features * 2, features * 4),
            self._block(features * 4, features * 8),
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.network(images).flatten()


class Generator(nn.Module):
    """DCGAN-style generator that maps latent vectors to 64x64 images."""

    def __init__(
        self, latent_dim: int = 100, image_channels: int = 1, features: int = 16
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            self._block(latent_dim, features * 16, stride=1, padding=0),
            self._block(features * 16, features * 8),
            self._block(features * 8, features * 4),
            self._block(features * 4, features * 2),
            nn.ConvTranspose2d(
                features * 2,
                image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    @staticmethod
    def _block(
        in_channels: int, out_channels: int, stride: int = 2, padding: int = 1
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, noise: Tensor) -> Tensor:
        return self.network(noise)


def initialize_weights(module: nn.Module) -> None:
    """Apply the normal initialization convention used in DCGAN."""

    for layer in module.modules():
        if isinstance(
            layer,
            (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.InstanceNorm2d),
        ):
            if layer.weight is not None:
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def gradient_penalty(
    critic: nn.Module, real: Tensor, fake: Tensor, *, device: torch.device
) -> Tensor:
    """Estimate the WGAN-GP unit-gradient penalty on random interpolations."""

    if real.shape != fake.shape:
        raise ValueError("real and fake batches must have identical shapes")
    batch_size = real.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real + (1.0 - alpha) * fake
    interpolated.requires_grad_(True)
    scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
    )[0]
    gradient_norm = gradients.flatten(start_dim=1).norm(2, dim=1)
    return ((gradient_norm - 1.0) ** 2).mean()
