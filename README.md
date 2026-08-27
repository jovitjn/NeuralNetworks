# Neural Networks from First Principles

A collection of compact PyTorch/Jupyter implementations built to understand how common neural-network architectures work internally rather than relying only on high-level training abstractions.

The repository covers the progression from basic feed-forward networks to convolutional models and several generations of GAN objectives.

## Included implementations

| Component | What it demonstrates |
| --- | --- |
| `basicNN.ipynb` | Core feed-forward neural-network workflow and training loop. |
| `CNN.ipynb` | Convolutional layers, feature extraction, pooling and image classification. |
| `basicGAN/` | Basic adversarial training with separate generator and discriminator networks. |
| `DC_GAN.ipynb` | Convolutional GAN architecture for image generation. |
| `WGAN.ipynb` | Wasserstein GAN training objective and critic-based optimization. |
| `WGAN_gradientpenalty.py` | WGAN-GP with an explicit gradient-penalty implementation. |

## WGAN-GP implementation

The standalone `WGAN_gradientpenalty.py` contains a complete training example with:

- convolutional generator and critic
- DCGAN-style weight initialization
- Wasserstein critic objective
- multiple critic updates per generator step
- gradient penalty
- MNIST data loading
- TensorBoard logging

The gradient penalty is computed on interpolated real/fake samples:

\[
\lambda\,\mathbb{E}_{\hat{x}}\left(\|\nabla_{\hat{x}}D(\hat{x})\|_2-1\right)^2,
\]

which encourages the critic to satisfy the Lipschitz constraint without weight clipping.

## Why this repository exists

This is deliberately an **educational implementation repository**, not a claim of a novel architecture. I use it to work through the mechanics behind models that later appear as building blocks in computer-vision and generative-model research.

That means the emphasis is on readable implementations of:

- forward passes
- loss functions
- training dynamics
- initialization
- optimization
- architectural differences between GAN variants

## Running the WGAN-GP example

Create an environment with PyTorch and the required utilities:

```bash
pip install torch torchvision tensorboard tqdm
```

Then run:

```bash
python WGAN_gradientpenalty.py
```

The script automatically downloads MNIST and writes real/fake image grids to TensorBoard logs.

To inspect them:

```bash
tensorboard --logdir logs
```

## Notes

Some notebooks contain saved outputs from the original experiments and are therefore larger than ideal. The code is being retained because it documents my progression through the underlying architectures; future cleanup can convert the strongest notebooks into smaller scripts/modules and strip embedded notebook outputs.

## References

The implementations are learning exercises based on the original ideas behind CNNs, GANs, DCGAN, WGAN and WGAN-GP. Where an implementation closely follows a paper/tutorial convention, it should be read as an educational reproduction rather than original research.
