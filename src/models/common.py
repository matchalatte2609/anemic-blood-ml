"""One training loop, shared by every deep arm.

This module exists for fairness rather than convenience. The easiest way to
manufacture a fusion win is to give the fusion model more epochs, a better
optimiser or a longer patience than the unimodal arms it is compared against --
the delta then measures the schedule, not the modality. Routing image, concat and
FiLM through a single function with one set of defaults makes that mistake
structurally difficult instead of merely discouraged.

If you tune anything here, every deep arm must be re-run.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.config import DEVICE, SEED

DEEP_DEFAULTS: dict = {
    "epochs": 25,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "patience": 6,
}


@torch.no_grad()
def predict(model, loader, device: str) -> np.ndarray:
    model.eval()
    out = []
    for img, tab, _ in loader:
        logits = model(img.to(device), tab.to(device) if tab.numel() else None)
        out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out)


def fit_deep(
    model: nn.Module,
    train_loader,
    es_loader,
    predict_loader,
    seed: int = SEED,
    device: str | None = None,
    verbose: bool = False,
    **overrides,
):
    """Fine-tune `model`, early-stopping on `es_loader`; return (model, preds).

    `es_loader` must be built from an inner split of the TRAINING rows, never the
    outer validation fold -- same rule the tabular arms follow. The outer fold is
    only ever passed as `predict_loader`, and its labels are never read here.

    Best-epoch weights are restored before predicting, so a late-training
    collapse cannot silently become the reported score.
    """
    cfg = {**DEEP_DEFAULTS, **overrides}
    device = device or DEVICE
    torch.manual_seed(seed)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    es_y = np.concatenate([y.numpy() for _, _, y in es_loader])
    best_auc, best_state, waited = -np.inf, None, 0

    for epoch in range(cfg["epochs"]):
        model.train()
        for img, tab, y in train_loader:
            img, y = img.to(device), y.to(device)
            tab = tab.to(device) if tab.numel() else None
            opt.zero_grad()
            loss_fn(model(img, tab), y).backward()
            opt.step()

        auc = roc_auc_score(es_y, predict(model, es_loader, device))
        if verbose:
            print(f"    epoch {epoch:2d}  inner-val auroc={auc:.3f}")

        if auc > best_auc:
            best_auc, waited = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, predict(model, predict_loader, device)
