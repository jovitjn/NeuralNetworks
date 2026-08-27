# Neural-network experiments

This repository is a record of how I learned the main neural-network building blocks in PyTorch, starting with a feed-forward network and CNN and then moving through GAN, DCGAN, WGAN, and WGAN-GP.

Most of the early work is kept in notebooks. The WGAN-GP experiment is the part I still use as normal Python code: `wgan_gp.py` defines the models and gradient penalty, while `WGAN_gradientpenalty.py` handles training, logging, samples, and checkpoints.

## What is here

| Path | Experiment |
|---|---|
| `basicNN.ipynb` | Feed-forward classification |
| `CNN.ipynb` | Convolution and pooling |
| `basicGAN/` | First GAN implementation |
| `DC_GAN.ipynb` | Convolutional generator and discriminator |
| `WGAN.ipynb` | Wasserstein critic objective |
| `wgan_gp.py` | Generator, critic, initialization, and gradient penalty |
| `WGAN_gradientpenalty.py` | Configurable WGAN-GP training script |

These are implementations of established architectures, not new model proposals.

## WGAN-GP objective

The critic loss is

\[
\mathcal{L}_D = \mathbb{E}[D(\tilde{x})]-\mathbb{E}[D(x)]
+\lambda\,\mathbb{E}\left(\lVert\nabla_{\hat{x}}D(\hat{x})\rVert_2-1\right)^2,
\]

and the generator minimizes \(\mathcal{L}_G=-\mathbb{E}[D(\tilde{x})]\).

The training script uses detached fake images for critic updates, five critic steps per generator step by default, deterministic seeding, TensorBoard logging, periodic image grids, and resumable model/optimizer checkpoints. It runs on CPU, CUDA, or Apple MPS.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train on MNIST:

```bash
python WGAN_gradientpenalty.py \
  --dataset mnist \
  --data-dir data \
  --epochs 100 \
  --batch-size 64 \
  --critic-steps 5 \
  --gradient-penalty-weight 10 \
  --seed 42
```

For a quick check of the data loader, both optimization steps, logging, sampling, and checkpoint writing:

```bash
python WGAN_gradientpenalty.py \
  --dataset mnist --epochs 1 --max-batches 2 --workers 0
```

For a custom dataset, use the folder structure expected by `torchvision.datasets.ImageFolder` and pass `--dataset image-folder --data-dir /path/to/data --image-channels 3`.

## Outputs

TensorBoard events and PNG sample grids are written below `logs/wgan_gp/`; checkpoints go to `checkpoints/`. A checkpoint contains both networks, both optimizers, the completed epoch, the global step, and the run configuration.

The model is fixed to 64x64 images. Seeds reduce run-to-run variation, but exact values can still differ across hardware and PyTorch versions. I have not included FID scores here; the repository is mainly about implementing and checking the training mechanics.