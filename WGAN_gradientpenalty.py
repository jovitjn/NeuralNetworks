"""Train a configurable WGAN-GP on MNIST or an image-folder dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from wgan_gp import Critic, Generator, gradient_penalty, initialize_weights


@dataclass(frozen=True)
class TrainConfig:
    dataset: str
    data_dir: Path
    image_channels: int
    epochs: int
    batch_size: int
    learning_rate: float
    latent_dim: int
    generator_features: int
    critic_features: int
    critic_steps: int
    gradient_penalty_weight: float
    workers: int
    seed: int
    log_dir: Path
    checkpoint_dir: Path
    checkpoint_every: int
    sample_every: int
    resume: Path | None
    device: str
    max_batches: int | None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "image-folder"), default="mnist")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--image-channels", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--generator-features", type=int, default=16)
    parser.add_argument("--critic-features", type=int, default=16)
    parser.add_argument("--critic-steps", type=int, default=5)
    parser.add_argument("--gradient-penalty-weight", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/wgan_gp"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--max-batches",
        type=int,
        help="optional per-epoch limit for smoke tests",
    )
    args = parser.parse_args()
    channels = args.image_channels or (1 if args.dataset == "mnist" else 3)
    config = TrainConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        image_channels=channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        generator_features=args.generator_features,
        critic_features=args.critic_features,
        critic_steps=args.critic_steps,
        gradient_penalty_weight=args.gradient_penalty_weight,
        workers=args.workers,
        seed=args.seed,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        sample_every=args.sample_every,
        resume=args.resume,
        device=args.device,
        max_batches=args.max_batches,
    )
    validate_config(config)
    return config


def validate_config(config: TrainConfig) -> None:
    positive_integers = {
        "image_channels": config.image_channels,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "latent_dim": config.latent_dim,
        "generator_features": config.generator_features,
        "critic_features": config.critic_features,
        "critic_steps": config.critic_steps,
        "checkpoint_every": config.checkpoint_every,
        "sample_every": config.sample_every,
    }
    for name, value in positive_integers.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if config.learning_rate <= 0 or config.gradient_penalty_weight < 0:
        raise ValueError("learning_rate must be positive and gradient penalty non-negative")
    if config.workers < 0:
        raise ValueError("workers cannot be negative")
    if config.max_batches is not None and config.max_batches < 1:
        raise ValueError("max_batches must be positive")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dataset(config: TrainConfig) -> Dataset:
    channel_transform = None
    if config.image_channels == 1:
        channel_transform = transforms.Grayscale(num_output_channels=1)
    elif config.image_channels != 3:
        raise ValueError("image_channels must be 1 or 3 for image datasets")

    steps = [transforms.Resize((64, 64))]
    if channel_transform is not None:
        steps.append(channel_transform)
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5] * config.image_channels, [0.5] * config.image_channels
            ),
        ]
    )
    transform = transforms.Compose(steps)
    if config.dataset == "mnist":
        return datasets.MNIST(
            root=config.data_dir, train=True, transform=transform, download=True
        )
    return datasets.ImageFolder(root=config.data_dir, transform=transform)


def create_loader(config: TrainConfig, dataset: Dataset) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.workers > 0,
        generator=generator,
    )


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    generator: Generator,
    critic: Critic,
    generator_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    generator: Generator,
    critic: Critic,
    generator_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    generator.load_state_dict(checkpoint["generator"])
    critic.load_state_dict(checkpoint["critic"])
    generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
    critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])


def sample_noise(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, latent_dim, 1, 1, device=device)


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = resolve_device(config.device)
    dataset = create_dataset(config)
    loader = create_loader(config, dataset)

    generator = Generator(
        config.latent_dim, config.image_channels, config.generator_features
    ).to(device)
    critic = Critic(config.image_channels, config.critic_features).to(device)
    initialize_weights(generator)
    initialize_weights(critic)
    generator_optimizer = Adam(
        generator.parameters(), lr=config.learning_rate, betas=(0.0, 0.9)
    )
    critic_optimizer = Adam(
        critic.parameters(), lr=config.learning_rate, betas=(0.0, 0.9)
    )

    start_epoch = 0
    global_step = 0
    if config.resume is not None:
        start_epoch, global_step = load_checkpoint(
            config.resume,
            generator=generator,
            critic=critic,
            generator_optimizer=generator_optimizer,
            critic_optimizer=critic_optimizer,
            device=device,
        )

    fixed_noise = sample_noise(32, config.latent_dim, device)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = config.log_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} dataset_size={len(dataset)} start_epoch={start_epoch}")

    with SummaryWriter(log_dir=config.log_dir) as writer:
        for epoch in range(start_epoch, config.epochs):
            progress = tqdm(loader, desc=f"epoch {epoch + 1}/{config.epochs}")
            for batch_index, (real, _) in enumerate(progress):
                if config.max_batches is not None and batch_index >= config.max_batches:
                    break
                real = real.to(device, non_blocking=True)
                batch_size = real.shape[0]

                set_requires_grad(critic, True)
                for _ in range(config.critic_steps):
                    with torch.no_grad():
                        fake = generator(sample_noise(batch_size, config.latent_dim, device))
                    critic_real = critic(real)
                    critic_fake = critic(fake)
                    penalty = gradient_penalty(critic, real, fake, device=device)
                    critic_loss = (
                        critic_fake.mean()
                        - critic_real.mean()
                        + config.gradient_penalty_weight * penalty
                    )
                    critic_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    critic_optimizer.step()

                set_requires_grad(critic, False)
                fake = generator(sample_noise(batch_size, config.latent_dim, device))
                generator_loss = -critic(fake).mean()
                generator_optimizer.zero_grad(set_to_none=True)
                generator_loss.backward()
                generator_optimizer.step()
                set_requires_grad(critic, True)

                writer.add_scalar("loss/critic", critic_loss.item(), global_step)
                writer.add_scalar("loss/generator", generator_loss.item(), global_step)
                writer.add_scalar("loss/gradient_penalty", penalty.item(), global_step)
                progress.set_postfix(
                    critic=f"{critic_loss.item():.3f}",
                    generator=f"{generator_loss.item():.3f}",
                )

                if global_step % config.sample_every == 0:
                    generator.eval()
                    with torch.inference_mode():
                        samples = generator(fixed_noise)
                    generator.train()
                    grid = make_grid(samples, normalize=True, value_range=(-1, 1))
                    writer.add_image("samples/generated", grid, global_step)
                    save_image(
                        samples,
                        sample_dir / f"step_{global_step:07d}.png",
                        normalize=True,
                        value_range=(-1, 1),
                    )
                global_step += 1

            if (epoch + 1) % config.checkpoint_every == 0:
                save_checkpoint(
                    config.checkpoint_dir / f"wgan_gp_epoch_{epoch + 1:03d}.pt",
                    epoch=epoch,
                    global_step=global_step,
                    generator=generator,
                    critic=critic,
                    generator_optimizer=generator_optimizer,
                    critic_optimizer=critic_optimizer,
                    config=config,
                )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
