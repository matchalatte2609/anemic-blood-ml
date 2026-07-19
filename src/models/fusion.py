"""Fusion arms: concatenation, and FiLM conditioning (the DAFT idea).

Concat first. FiLM is only worth running if concat already beats both unimodal
arms -- otherwise you are tuning a second fusion variant to rescue a null result,
which is exactly the practice this project exists to avoid.

Both take `(img, tab)` and return a logit, matching `ImageEncoder`, so all three
deep arms share `models.common.fit_deep` and therefore an identical budget.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.image import ImageEncoder
from src.models.tabular import TabularMLP


class ConcatFusion(nn.Module):
    """Image embedding (+) tabular embedding -> MLP head.

    The projection layers are the point. A ResNet18 embedding is 512-d and the
    tabular one is 64-d; concatenated raw, the head sees 8x more image than
    tabular and can reach a good solution while ignoring the CBC entirely -- the
    "fusion" then quietly degenerates into the image arm, and its delta measures
    nothing. Projecting both to `proj_dim` first forces an even hearing.
    """

    def __init__(
        self,
        image_encoder: ImageEncoder,
        tabular_encoder: TabularMLP,
        proj_dim: int = 64,
        hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.image = image_encoder
        self.tabular = tabular_encoder
        self.img_proj = nn.Linear(image_encoder.out_dim, proj_dim)
        self.tab_proj = nn.Linear(tabular_encoder.out_dim, proj_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * proj_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, img: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        z_img = self.img_proj(self.image.forward_features(img))
        z_tab = self.tab_proj(self.tabular.forward_features(tab))
        return self.head(torch.cat([z_img, z_tab], dim=1)).squeeze(-1)


class FiLMLayer(nn.Module):
    """Per-channel scale and shift predicted from the tabular vector.

        out = (1 + gamma(tab)) * feature_map + beta(tab)

    broadcast over H and W, so the CBC decides which visual channels to amplify
    or suppress rather than being appended as extra evidence.

    The final linear layer is zero-initialised, which makes gamma = beta = 0 at
    step 0 and the whole layer an exact identity. So an untrained FiLM model *is*
    the image model, and any measured difference is attributable to conditioning
    rather than to a randomly perturbed starting point.
    """

    def __init__(self, n_channels: int, tab_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tab_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * n_channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature_map: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(tab).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feature_map * (1 + gamma) + beta


class FiLMFusion(nn.Module):
    """Image backbone with FiLM applied to the final feature map.

    Conditioning goes late, not at the stem. Early conditioning hands the tabular
    vector control over low-level filters, which at 800 training images mostly
    buys a harder optimisation problem.
    """

    def __init__(self, tab_dim: int, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.encoder = ImageEncoder(backbone, pretrained)
        self.film = FiLMLayer(self.encoder.out_dim, tab_dim)
        self.head = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, img: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        fmap = self.film(self.encoder.feature_map(img), tab)
        return self.head(fmap.mean(dim=(2, 3))).squeeze(-1)
