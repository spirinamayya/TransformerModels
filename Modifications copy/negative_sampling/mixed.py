from rectools.models.nn.transformers.negative_sampler import TransformerNegativeSamplerBase, CatalogUniformSampler
from rectools.models.nn.transformers.lightning import TransformerLightningModule
import typing as tp
from collections.abc import Hashable
from rectools import Columns

import torch

from rectools import ExternalIds
from rectools.dataset.dataset import DatasetSchemaDict

from rectools.models.nn.transformers.data_preparator import TransformerDataPreparatorBase
from rectools.models.nn.transformers.torch_backbone import TransformerBackboneBase
from .in_batch import InBatchSampler



MIXED_KWARGS = {"ratio": 0.5}

class MixedSampler(TransformerNegativeSamplerBase):
    """Class to sample negatives uniformly from all catalog items."""
    def __init__(
        self,
        n_negatives: int,
        ratio: float = 0.5,
        **kwargs: tp.Any,
    ) -> None:
        super().__init__(n_negatives)
        self.in_batch_sampler = InBatchSampler(int(n_negatives * ratio))
        self.uniform_sampler = CatalogUniformSampler(n_negatives - int(n_negatives * ratio))
        self.ratio = ratio

    def get_negatives(self, batch_dict: tp.Dict, lowest_id: int, highest_id: int, session_len_limit: tp.Optional[int] = None) -> torch.Tensor:
        """Return sampled negatives."""
        in_batch_negatives = self.in_batch_sampler.get_negatives(batch_dict, lowest_id, highest_id, session_len_limit)
        uniform_negatives = self.uniform_sampler.get_negatives(batch_dict, lowest_id, highest_id, session_len_limit)
        return torch.cat([in_batch_negatives, uniform_negatives], dim=-1)

