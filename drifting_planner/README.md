# Drifting Planner

One-step trajectory generation using Drifting Models (Deng et al., 2025).

## Overview

This package implements the Drifting Model for autonomous driving trajectory generation. Unlike diffusion models that require iterative denoising at inference time, the Drifting Model learns to generate trajectories in a single forward pass by evolving the pushforward distribution during training.

## Architecture

- **Encoder**: Scene context encoder (ego history, neighbors, lanes, route, etc.)
- **Decoder**: DiT-based trajectory generator with one-step generation
- **Drifting Loss**: Anti-symmetric mean-shift field loss that attracts generated samples toward data and repels from generated distribution

## Training

```bash
cd drifting_planner
bash train_run.sh <exp_name> <train_set_list> <valid_set_list>
```

## Key Differences from Diffusion Planner

| Aspect | Diffusion Planner | Drifting Planner |
|--------|------------------|------------------|
| Inference | Multi-step (10+ NFE) | One-step (1 NFE) |
| Training loss | Diffusion/flow matching | Drifting field + prediction |
| Sampling | Iterative denoising | Single forward pass |

## Package Structure

```
drifting_planner/
├── train_drifting.py          # Main training script
├── train_run.sh               # Training launcher
├── pyproject.toml             # Package config
└── drifting_planner/
    ├── dimensions.py          # Copied from diffusion_planner
    ├── loss.py                # Loss functions
    ├── drifting_loss.py       # Drifting field computation
    ├── train_epoch.py         # Training loop
    ├── model/
    │   ├── drifting_planner.py  # Main model
    │   └── module/
    │       ├── encoder.py       # Scene encoder
    │       ├── decoder.py       # Trajectory decoder
    │       ├── dit.py           # DiT architecture
    │       └── mixer.py         # Mixer blocks
    └── utils/                 # Utilities (normalizer, dataset, etc.)
```
