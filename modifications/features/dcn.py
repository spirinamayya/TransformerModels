import typing as tp

import pandas as pd
import torch
import typing_extensions as tpe
from torch import nn

from rectools.dataset.dataset import Dataset, DatasetSchema, SparseFeaturesSchema
from rectools.dataset.features import SparseFeatures
from rectools.models.nn.item_net import CatFeaturesItemNet, ItemNetBase, ItemNetConstructorBase
from rectools.models.similarity.base import SimilarityModuleBase


class CatFeaturesStackedItemNet(ItemNetBase):
    """Network for item embeddings that computes embeddings for each feature and stacks them."""

    def __init__(
        self,
        emb_bag_inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        offsets: torch.Tensor,
        category_cardinalities: torch.Tensor,
        emb_bag_category_inds: torch.Tensor,
        n_items: int,
        dropout_rate: float,
        **kwargs: tp.Any,
    ):
        super().__init__()

        self.category_embeddings = nn.ModuleList(
            nn.EmbeddingBag(num_embeddings=card.item(), embedding_dim=int(6 * card ** (1 / 4)))
            for card in category_cardinalities
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.n_items = n_items

        self.register_buffer("offsets", offsets)
        self.register_buffer("emb_bag_inputs", emb_bag_inputs)
        self.register_buffer("input_lengths", input_lengths)
        self.register_buffer("emb_bag_category_inds", emb_bag_category_inds)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """Forward pass to get stacked item embeddings from categorical item features."""
        item_embeddings = []
        for ind, embedding_bag in enumerate(self.category_embeddings):
            item_emb_bag_inputs, item_offsets = self._get_item_inputs_offsets(items, ind)
            feature_embeddings_per_items = embedding_bag(input=item_emb_bag_inputs, offsets=item_offsets)
            item_embeddings.append(feature_embeddings_per_items)

        concat_item_embs = torch.cat(item_embeddings, dim=-1)
        concat_item_embs = self.dropout(concat_item_embs)
        return concat_item_embs

    def _get_item_inputs_offsets(self, items: torch.Tensor, ind: int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        embedding_slice = slice(
            self.get_buffer("emb_bag_category_inds")[ind], self.get_buffer("emb_bag_category_inds")[ind + 1]
        )
        offset_slice = slice(self.n_items * ind, self.n_items * (ind + 1))
        length_range = torch.arange(self.get_buffer("input_lengths")[offset_slice].max().item(), device=self.device)
        item_indexes = self.get_buffer("offsets")[offset_slice][items].unsqueeze(-1) + length_range
        length_mask = length_range < self.get_buffer("input_lengths")[offset_slice][items].unsqueeze(-1)
        item_emb_bag_inputs = self.get_buffer("emb_bag_inputs")[embedding_slice][item_indexes[length_mask]]
        item_offsets = torch.cat(
            (
                torch.tensor([0], device=self.device),
                torch.cumsum(self.get_buffer("input_lengths")[offset_slice][items], dim=0)[:-1],
            )
        )
        return item_emb_bag_inputs, item_offsets

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        n_factors: int,
        dropout_rate: float,
    ) -> tp.Optional[tpe.Self]:
        """Create CatFeaturesStackedItemNet from RecTools dataset."""
        dataset_schema = DatasetSchema.model_validate(dataset.get_schema())
        CatFeaturesItemNet._warn_for_unsupported_dataset_schema(dataset_schema)  # pylint: disable=protected-access

        if isinstance(dataset.item_features, SparseFeatures):
            item_cat_features = dataset.item_features.get_cat_features()
            if item_cat_features.values.size == 0:
                return None

            category_cardinalities = torch.tensor(
                pd.DataFrame(item_cat_features.names).groupby(0, sort=False)[1].nunique().values
            )
            category_inds = torch.cumsum(torch.cat((torch.tensor([0]), category_cardinalities)), dim=0)

            emb_bag_inputs, offsets, input_lengths, category_lengths = ([] for _ in range(4))
            for _, (current_index, next_index) in enumerate(zip(category_inds, category_inds[1:])):
                category_features = item_cat_features.values[:, current_index:next_index]
                emb_bag_inputs.append(torch.tensor(category_features.indices))
                category_offsets = torch.tensor(category_features.indptr)
                offsets.append(category_offsets[:-1])
                input_lengths.append(torch.diff(category_offsets, dim=0))
                category_lengths.append(len(category_features.data))

            emb_bag_category_inds = torch.cumsum(torch.cat((torch.tensor([0]), torch.tensor(category_lengths))), dim=0)

            return cls(
                emb_bag_inputs=torch.cat(emb_bag_inputs),
                offsets=torch.cat(offsets),
                input_lengths=torch.cat(input_lengths),
                category_cardinalities=category_cardinalities,
                emb_bag_category_inds=emb_bag_category_inds,
                n_items=dataset_schema.items.n_hot,
                dropout_rate=dropout_rate,
            )
        return None

    @classmethod
    def from_dataset_schema(
        cls,
        dataset_schema: DatasetSchema,
        n_factors: int,
        dropout_rate: float,
        **kwargs: tp.Any,
    ) -> tp.Optional[tpe.Self]:
        """Construct CatFeaturesStackedItemNet from Dataset schema."""
        CatFeaturesItemNet._warn_for_unsupported_dataset_schema(dataset_schema)  # pylint: disable=protected-access
        features_schema = dataset_schema.items.features

        if isinstance(features_schema, SparseFeaturesSchema) and len(features_schema.cat_feature_indices) > 0:
            category_cardinalities = torch.tensor(
                pd.DataFrame(features_schema.names)
                .iloc[features_schema.cat_feature_indices]
                .groupby(0, sort=False)[1]
                .nunique()
                .values
            )
            n_categories = category_cardinalities.shape[0]

            emb_bag_inputs = torch.randint(high=dataset_schema.items.n_hot, size=(features_schema.cat_n_stored_values,))
            offsets = torch.randint(high=dataset_schema.items.n_hot, size=(dataset_schema.items.n_hot * n_categories,))
            input_lengths = torch.randint(
                high=dataset_schema.items.n_hot, size=(dataset_schema.items.n_hot * n_categories,)
            )
            emb_bag_category_inds = torch.randint(high=dataset_schema.items.n_hot, size=(n_categories + 1,))
            return cls(
                emb_bag_inputs=emb_bag_inputs,
                input_lengths=input_lengths,
                offsets=offsets,
                category_cardinalities=category_cardinalities,
                emb_bag_category_inds=emb_bag_category_inds,
                n_items=dataset_schema.items.n_hot,
                dropout_rate=dropout_rate,
            )
        return None

    @property
    def out_dim(self) -> int:
        """Return categorical item embedding output dimension."""
        out_dim = 0
        for embedding in self.category_embeddings:
            out_dim += embedding.embedding_dim
        return out_dim


class DCNCrossNetwork(nn.Module):
    """Cross Network from https://arxiv.org/pdf/1708.05123"""

    def __init__(
        self,
        in_features: int,
        n_cross_layers: int,
    ):
        super().__init__()
        self.n_cross_layers = n_cross_layers
        self.cross_layer_weights = nn.Parameter(torch.rand(n_cross_layers, in_features, 1))  # in DCNV2 not 1
        self.cross_layer_biases = nn.Parameter(torch.rand(n_cross_layers, in_features, 1))

    def forward(
        self,
        embs: torch.Tensor,  # [batch_size, session_max_len * n_factors]
    ) -> torch.Tensor:
        """Forward pass to compute version 1 cross network embeddings."""
        embs_0 = embs.clone().detach().unsqueeze(2)
        current_embs = embs.unsqueeze(2)

        for layer_idx in range(self.n_cross_layers):
            feature_crossing = embs_0 @ current_embs.transpose(1, 2) @ self.cross_layer_weights[layer_idx]
            current_embs = feature_crossing + self.cross_layer_biases[layer_idx] + current_embs
        current_embs = current_embs.squeeze(2)
        return current_embs


class DCNv2CrossNetwork(nn.Module):
    """Cross Network from https://arxiv.org/pdf/2008.13535"""

    def __init__(
        self,
        in_features: int,
        n_cross_layers: int,
    ):
        super().__init__()
        self.n_cross_layers = n_cross_layers
        self.cross_layer_weights = nn.Parameter(torch.rand(n_cross_layers, in_features, in_features))
        self.cross_layer_biases = nn.Parameter(torch.rand(n_cross_layers, in_features, 1))

    def forward(
        self,
        embs: torch.Tensor,  # [batch_size, session_max_len * n_factors]
    ) -> torch.Tensor:
        """Forward pass to compute version 2 cross network embeddings."""
        embs_0 = embs.clone().detach().unsqueeze(2)
        current_embs = embs.unsqueeze(2)

        for layer_idx in range(self.n_cross_layers):
            # TODO: check formula
            feature_crossing = self.cross_layer_weights[layer_idx] @ current_embs + self.cross_layer_biases[layer_idx]
            current_embs = embs_0 * feature_crossing + current_embs
        current_embs = current_embs.squeeze(2)
        return current_embs


class DeepNetwork(nn.Module):
    """Deep network."""

    def __init__(
        self,
        in_features: int,
        deep_layers_dim: list,
        dropout_rate: float,
    ):
        super().__init__()

        deep_layers_dim = [in_features] + deep_layers_dim
        self.linear_layers = nn.ModuleList(
            [nn.Linear(deep_layers_dim[i], deep_layers_dim[i + 1]) for i in range(len(deep_layers_dim) - 1)]
        )
        self.batch_norm = nn.ModuleList(
            [nn.BatchNorm1d(deep_layers_dim[i + 1]) for i in range(len(deep_layers_dim) - 1)]
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        embs: torch.Tensor,  # [batch_size, session_max_len * n_factors]
    ) -> torch.Tensor:
        """Forward pass to compute embeddings using deep network."""
        for layer_idx, layer in enumerate(self.linear_layers):
            embs = self.activation(self.batch_norm[layer_idx](layer(embs)))
            embs = self.dropout(embs)
        return embs


class DCNNetworkConstructor(nn.Module):
    """
    Construct DCN network based on version and format.

    Parameters
    ----------
    n_cross_layers : int
        Number of cross layers.
    deep_layers_dim : List
        List of hidden units for deep network.
    dcn_version : {1, 2}
        Version of cross network.
    dcn_form : {"parallel", "stacked"}
        Way to compute deep and cross network. If "parallel", compute networks separately and stack results.
        If "stacked", embeddings are passed through deep network first and after that through cross network.
    dropout_rate: float
        Dropout rate for deep network.
    in_features: int
        Initial dimension of item embeddings.
    """

    def __init__(
        self,
        n_cross_layers: int,
        deep_layers_dim: tp.List,
        dcn_version: int,
        dcn_form: str,
        dropout_rate: float,
        in_features: int,
    ) -> None:
        super().__init__()
        self.n_cross_layers = n_cross_layers
        self.deep_layers_dim = deep_layers_dim
        self.in_features = in_features

        self.dcn_layers = nn.ModuleList()
        if n_cross_layers > 0:
            if dcn_version == 1:
                self.dcn_layers.add_module(
                    "cross_network", DCNCrossNetwork(in_features=in_features, n_cross_layers=n_cross_layers)
                )
            elif dcn_version == 2:
                self.dcn_layers.add_module(
                    "cross_network", DCNv2CrossNetwork(in_features=in_features, n_cross_layers=n_cross_layers)
                )
        if len(self.deep_layers_dim) > 0:
            self.dcn_layers.add_module(
                "deep_network",
                DeepNetwork(in_features=in_features, deep_layers_dim=deep_layers_dim, dropout_rate=dropout_rate),
            )
        if len(self.dcn_layers) == 0:
            raise ValueError("No cross or deep layers")

        self.dcn_form = dcn_form
        if dcn_form == "stacked":
            self.forward_func = self.stacked_forward
        elif dcn_form == "parallel":
            self.forward_func = self.parallel_forward
        else:
            raise ValueError("No cross or deep layers")

    def parallel_forward(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass for parallel DCN structure."""
        layer_embeddings = []
        for layer in self.dcn_layers:
            layer_embeddings.append(layer(item_embeddings))
        output = torch.cat((layer_embeddings), dim=1)
        return output

    def stacked_forward(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass for stacked DCN structure."""
        for layer in self.dcn_layers:
            item_embeddings = layer(item_embeddings)
        return item_embeddings

    def forward(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.forward_func(item_embeddings)

    @property
    def out_dim(self) -> int:
        """Return output dimension of dcn network."""
        if self.dcn_form == "stacked":
            if len(self.deep_layers_dim) > 0:
                return self.deep_layers_dim[-1]
            if self.n_cross_layers > 0:
                return self.in_features
        if self.dcn_form == "parallel":
            in_features = 0
            if self.n_cross_layers > 0:
                in_features += self.in_features
            if len(self.deep_layers_dim) > 0:
                in_features += self.deep_layers_dim[-1]
            return in_features
        raise ValueError("only stacked and parallel dcn forms are supported.")


class StackedEmbeddingsConstructor(ItemNetConstructorBase):
    """Item net blocks constructor that concatenates all of the its net blocks embeddings."""

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through item net blocks and aggregation of the results.
        Concatenation of embeddings.
        """
        item_embs = []
        for idx_block in range(self.n_item_blocks):
            item_emb = self.item_net_blocks[idx_block](items)
            item_embs.append(item_emb)
        return torch.cat(item_embs, dim=-1)

    @property
    def out_dim(self) -> int:
        """Return item net constructor output dimension."""
        out_dim = 0
        for _, block in enumerate(self.item_net_blocks):
            out_dim += block.out_dim
        return out_dim


class DCNEmbeddingConstructor(ItemNetConstructorBase):
    """Embedding constructor that concatenates item net blocks and applies DCN network.

    Parameters
    ----------
    n_items: int,
        Number of items in the dataset.
    item_net_blocks : Sequence(ItemNetBase)
        Latent embedding size of item embeddings.
    n_cross_layers : int
        Number of cross layers.
    deep_layers_dim : List
        List of hidden dimensions for deep network.
    n_factors: int
        Latent embedding size of item embeddings.
    dcn_version : {1, 2}
        Version of cross network.
    dcn_form : {"parallel", "stacked"}
        Way to compute deep and cross network. If "parallel", compute networks separately and stack results.
        If "stacked", embeddings are passed through deep network first and after that through cross network.
    dropout_rate: float
        Dropout rate for deep network.
    """

    def __init__(
        self,
        n_items: int,
        item_net_blocks: tp.Sequence[ItemNetBase],
        n_cross_layers: int,
        deep_layers_dim: tp.List,
        n_factors: int,
        dcn_version: int,
        dcn_form: str,
        dropout_rate: float,
    ) -> None:
        super().__init__(
            n_items=n_items,
            item_net_blocks=item_net_blocks,
        )
        in_features = 0
        for _, block in enumerate(self.item_net_blocks):
            in_features += block.out_dim

        self.dcn_network = DCNNetworkConstructor(
            n_cross_layers,
            deep_layers_dim,
            dcn_version,
            dcn_form,
            dropout_rate,
            in_features=in_features,
        )
        self.linear_layer = nn.Linear(in_features=self.dcn_network.out_dim, out_features=n_factors)

    def _get_stacked_item_embeddings(self, items: torch.Tensor) -> torch.Tensor:
        """
        Pass through item net blocks and aggregation of the results.
        Conctatenation of embeddings.
        """
        item_embs = []
        for idx_block in range(self.n_item_blocks):
            item_emb = self.item_net_blocks[idx_block](items)
            item_embs.append(item_emb)
        return torch.cat(item_embs, dim=-1)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Concatenate net block embeddings.
        Pass embeddings through dcn network.
        Apply linear layer to get output dimension equal to ```n_factors```.
        """
        item_embeddings = self._get_stacked_item_embeddings(items)
        output = self.dcn_network(item_embeddings)
        output = self.linear_layer(output)
        return output

    @classmethod
    def from_dataset(  # type: ignore[override]
        cls,
        dataset: Dataset,
        n_factors: int,
        dropout_rate: float,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]],
        n_cross_layers: int,
        deep_layers_dim: tp.List[int],
        dcn_version: int,
        dcn_format: str,
        item_nets_factors: tp.Optional[int] = None,
        **kwargs: tp.Any,
    ) -> tpe.Self:
        """Create DCNEmbeddingConstructor from RecTools dataset."""
        n_items = dataset.item_id_map.size

        item_net_blocks: tp.List[ItemNetBase] = []
        for item_net in item_net_block_types:
            item_net_block = item_net.from_dataset(
                dataset=dataset,
                n_factors=n_factors if item_nets_factors is None else item_nets_factors,
                dropout_rate=dropout_rate,
            )
            if item_net_block is not None:
                item_net_blocks.append(item_net_block)

        return cls(
            n_items,
            item_net_blocks,
            n_cross_layers,
            deep_layers_dim,
            n_factors,
            dcn_version,
            dcn_format,
            dropout_rate,
        )

    @classmethod
    def from_dataset_schema(  # type: ignore[override]
        cls,
        dataset_schema: DatasetSchema,
        n_factors: int,
        dropout_rate: float,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]],
        n_cross_layers: int,
        deep_layers_dim: tp.List[int],
        dcn_version: int,
        dcn_format: str,
        item_nets_factors: tp.Optional[int] = None,
        **kwargs: tp.Any,
    ) -> tpe.Self:
        """Construct DCNEmbeddingConstructor from Dataset schema."""
        n_items = dataset_schema.items.n_hot

        item_net_blocks: tp.List[ItemNetBase] = []
        for item_net in item_net_block_types:
            item_net_block = item_net.from_dataset_schema(
                dataset_schema,
                n_factors if item_nets_factors is None else item_nets_factors,
                dropout_rate,
            )
            if item_net_block is not None:
                item_net_blocks.append(item_net_block)

        return cls(
            n_items,
            item_net_blocks,
            n_cross_layers,
            deep_layers_dim,
            n_factors,
            dcn_version,
            dcn_format,
            dropout_rate,
        )

    @property
    def out_dim(self) -> int:
        """Return output dimension of DCNEmbeddingConstructor network."""
        return self.dcn_network.out_dim
    
    self.dcn_network = DCNNetworkConstructor(
            n_cross_layers,
            deep_layers_dim,
            dcn_version,
            dcn_form,
            dropout_rate,
            in_features=in_features,
        )
