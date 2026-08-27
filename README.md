# Neural Networks from First Principles

Educational PyTorch implementations tracing a progression from feed-forward and convolutional networks to GAN, DCGAN, WGAN, and WGAN-GP objectives.

These are learning implementations and coursework experiments, not novel architectures or research results. The aim is to make the model components, objectives, and optimization mechanics explicit before using them in larger computer-vision and generative-model projects.

## Contents

| Component | What it demonstrates | Format |
|---|---|---|
| `basicNN.ipynb` | Feed-forward model and training loop | Original notebook |
| `CNN.ipynb` | Convolution, pooling, and image classification | Original notebook |
| `basicGAN/` | Basic generator/discriminator training | Notebook + script |
| `DC_GAN.ipynb` | Convolutional image generation | Original notebook |
| `WGAN.ipynb` | Wasserstein critic objective | Original notebook |
| `wgan_gp.py` | Reusable 64×64 generator, critic, initialization, and penalty | Python module |
| `WGAN_gradientpenalty.py` | Configurable WGAN-GP training, logging, sampling, and checkpoints | CLI script |

The historical notebooks are retained as the original experiments. The WGAN-GP implementation is the polished, reusable surface of this repository.

## WGAN-GP

The critic is trained to minimize

\[
\mathcal{L}_D = \mathbb{E}_{\tilde{x}\sim P_g}[D(\tilde{x})]
- \mathbb{E}_{x\sim P_r}[D(x)]
+ \lambda\,\mathbb{E}_{\hat{x}}
\left(\lVert\nabla_{\hat{x}}D(\hat{x})\rVert_2-1\right)^2,
\]

while the generator minimizes

\[
\mathcal{L}_G=-\mathbb{E}_{\tilde{x}\sim P_g}[D(\tilde{x})].
\]

`wgan_gp.py` provides:

- a DCGAN-style generator that maps latent vectors to 64×64 images
- an unconstrained convolutional `Critic` rather than a sigmoid discriminator
- DCGAN-style weight initialization
- gradient penalty on random real/fake interpolations

The training script adds:

- `argparse` configuration for the dataset and hyperparameters
- deterministic seed handling
- MNIST or `ImageFolder` input
- five critic steps per generator step by default
- detached generator outputs during critic optimization
- no unnecessary retained autograd graph
- TensorBoard losses and generated samples
- PNG sample grids
- resumable model/optimizer checkpoints
- automatic CPU, CUDA, or Apple MPS selection
- a `--max-batches` smoke-test option

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train on MNIST

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

The script resizes MNIST to 64×64, normalizes images to \([-1,1]\), downloads the dataset into `data/`, writes events and sample PNGs under `logs/wgan_gp/`, and stores checkpoints under `checkpoints/`.

For a quick pipeline check:

```bash
python WGAN_gradientpenalty.py --epochs 1 --max-batches 2 --workers 0
```

## Train on an image folder

Use the directory layout expected by `torchvision.datasets.ImageFolder`:

```text
data/celeba/
└── images/
    ├── image_0001.jpg
    └── image_0002.jpg
```

Then run:

```bash
python WGAN_gradientpenalty.py \
  --dataset image-folder \
  --data-dir data/celeba \
  --image-channels 3
```

## Resume a trusted checkpoint

```bash
python WGAN_gradientpenalty.py \
  --epochs 100 \
  --resume checkpoints/wgan_gp_epoch_020.pt
```

Checkpoints include both networks, both optimizers, the completed epoch, the global step, and the run configuration. Resume only checkpoints you created or otherwise trust.

## Reproducibility and limitations

- Seeds and deterministic cuDNN settings reduce nondeterminism, but exact results can still differ across devices and library versions.
- The convolutional architecture is specifically designed for 64×64 images.
- No quantitative generative metric such as FID is reported in this repository; do not infer image quality from the presence of a complete training pipeline.
- The default feature width is deliberately small for educational runs rather than tuned for best generation quality.
- Saved notebook outputs make some historical files large. They remain to preserve the original learning record and are not presented as the primary code path.

## Provenance

The implementations reproduce standard concepts from GAN, DCGAN, WGAN, and WGAN-GP literature and common instructional conventions. Architectural ideas and objectives belong to their original authors; this repository documents my implementation and learning process.
