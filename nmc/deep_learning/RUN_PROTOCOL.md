# OCV-start full baseline and feature ablation

This folder is an isolated run copy derived from `/home/user/바탕화면/DL/CEMA_MLP`.

## Data

- Label protocol: OCV-start SOC relabel.
- Raw root: `/home/user/바탕화면/DL/CEMA_MLP_OCVSTART_FULLGRID/nmc_ocvstart_lopo_clean`.
- The raw root is a symlink to the validated OCV-start clean data folder from the original workspace.
- Profiles: `DST`, `FUDS`, `US06`.
- Temperatures: `0C`, `25C`, `45C`.

## Split

- LOPO by profile.
- Holdout order: `US06`, `DST`, `FUDS`.
- Train profiles are the other two profiles.
- No validation profile is used: `valid_profiles=NONE`.
- Final epoch only is selected: `stage1_selector=last_epoch`, `fixed_stage1_epoch=epochs`.

## Training

- Seeds: `0,1,2`.
- Epochs: `200`.
- Batch size: `2048`.
- Window length: default copied config, `50`.
- Stride: default copied config, `3`.
- Stage 2: skipped.
- Final weights: saved.
- Predictions: saved.
- Train final evaluation: skipped.

## Grid

- The grid is generated over three holdouts.
- Heads: normal and residual.
- Existing model, feature, and training code is copied from the original folder and not reimplemented.
