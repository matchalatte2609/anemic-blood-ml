"""Image arm: ImageNet-pretrained CNN on the smear.

Recall from README.md that the label was produced by a pathologist looking at
this image. Unlike the usual setup, the image is the modality with the direct
path to the label and the CBC is the approximation. Expect this arm to be
competitive or better -- if it lands near chance, suspect the training loop
rather than concluding "vision doesn't work at n=1000".

Every deep model in this project takes `(img, tab)` and returns a logit, even
when it ignores one of them. That uniform signature lets all deep arms share one
training loop (`models.common.fit_deep`), which is what guarantees they get
identical epoch budgets and optimisers -- otherwise a fusion "win" could just be
a longer schedule.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """Pretrained backbone with the classifier stripped.

    `num_classes=0` makes timm return pooled features instead of logits, so the
    same object serves the image-only arm (via `forward`) and the fusion arms
    (via `forward_features` / `feature_map`).

    resnet18 by default. Do not reach for something bigger: 1000 images will not
    support it, and a heavier backbone mostly buys a more confident overfit.
    """

    def __init__(self, name: str = "resnet18", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.out_dim = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.out_dim, 1)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-pooling feature map, (B, C, H, W). FiLM conditions on this."""
        return self.backbone.forward_features(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled embedding, (B, out_dim). What concat fusion consumes."""
        return self.backbone(x)

    def forward(self, img: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.dropout(self.backbone(img))).squeeze(-1)


def pool_crop_embeddings(embeddings: torch.Tensor, parent_ids) -> torch.Tensor:
    """Mean-pool AneRBC-II crop embeddings up to one vector per parent.

    Not used by the default pipeline, which predicts on AneRBC-I parents so that
    every arm shares one prediction unit and the folds stay directly comparable.

    Implement this only if you decide the image arm needs the 12x extra training
    signal from the crops. If you do: the group key stays `{parent_id}_{cls}` and
    folds stay parent-level, or trap 2 bites.
    """
    raise NotImplementedError
