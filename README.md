# Alice: An Interpretable Neural Architecture for Generalization in Substitution Ciphers

Alice is a transformer-based neural network designed to solve cryptograms.

## Installation

### From Source

```bash
git clone https://github.com/al-jshen/alice.git
cd alice
pip install -e .
```

## Quick Start

### Training a Model

1. **Prepare your configuration**: Copy and modify one of the config files in `configs/`:

```bash
cp configs/alice_base.yaml configs/my_config.yaml
```

2. **Update the configuration**: Edit `configs/my_config.yaml` to set your dataset path, W&B credentials, etc.

3. **Train the model**:

```bash
python alice/train.py --config configs/my_config.yaml
```

### Pretrained Models

We release the checkpoints for various pretrained models in [`checkpoints/`](checkpoints/).

You will need to instantiate the model with the same configuration used during training.

## Model Variants

### Bijective Decoding

To use the bijective decoding head:

```yaml
sinkhorn_decoding: true
sinkhorn_iters: 10
sinkhorn_tau: 1.0
```

### Dynamic Embeddings

Enable dynamic embeddings:

```yaml
dynamic_embeddings: true
embedding_n_layer: 2
embedding_n_head: 4
```

## License

This project is licensed under the Apache-2.0 License. See [LICENSE](LICENSE) for details.
