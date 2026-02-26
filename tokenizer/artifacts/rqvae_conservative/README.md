# RQ-VAE Conservative Weights

This directory contains the best conservative RQ-VAE checkpoint exported for inference/evaluation use.

- `rqvae_best_infer.pt`: model weights + config + metrics (no optimizer/scheduler state)
- `manifest.json`: provenance, checksum, and size metadata

Load path example:

```python
from pathlib import Path
from animind_tokenizer.rqvae import load_rqvae_for_eval

model, payload = load_rqvae_for_eval(
    checkpoint_path=Path('artifacts/rqvae_conservative/rqvae_best_infer.pt'),
    device='cpu',
)
```
