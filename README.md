# DDTG_Net

This directory is a clean, standalone model-code release for the paper
*A Domain-Decomposition Green-Aware Transformer for Efficient Prediction of
Electromagnetic Scattering from 3-D PEC Targets*.

## Repository layout

```text
ddtg/
  features.py                    Paper-aligned 14-D facet features
  models/
    DDTGmodel.py                 Main DDGT-Net implementation
    baselines/
      gnnsolver.py               GNNsolver baseline
      pdegatscl.py               PdEgatSCL baseline
    ablations/
      factory.py                 Paper ablation factory
      patch_egat.py              Patch-EGAT ablation
```


## Main model

```python
from ddtg import DDTGmodel

model = DDTGmodel()
```

The defaults reproduce the architecture stated in the paper: 256 hidden
channels, six tokens per patch, four dense Green-aware patch Transformer
layers, and eight attention heads. The model returns four local tangential
coefficients `[Re(alpha), Im(alpha), Re(beta), Im(beta)]` per facet.

## Baselines and ablations

```python
from ddtg.models.baselines import GNNSolver, PdEgatSCL_model
from ddtg.models.ablations import build_ablation

gnnsolver = GNNSolver()
pdegatscl = PdEgatSCL_model()

full_model = build_ablation("full")
geometric_bias = build_ablation("geometric_bias")
without_green_bias = build_ablation("without_green_bias")
sparse_model = build_ablation("sparse")
patch_egat = build_ablation("patch_egat")
```


## Installation

```bash
pip install -r requirements.txt
```

Before publishing, add the license selected by the authors and update the
paper citation after the final bibliographic information becomes available.
