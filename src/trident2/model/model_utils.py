from collections import defaultdict
import json
import os
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Union
import torch.nn.functional as F
import numpy as np
import random
from loguru import logger
from ete3 import Tree


def load_taxonomic_embedding_dict(
    embedding_lookup="./data/taxid2embedding_dist2LCA=False_ambig=max_rank2dist=False_dim=768_metric=True_maxiter=10000_eps=NaN_ninit=1_2026-04-17.json",
) -> dict:
    if not os.path.exists(embedding_lookup):
        raise FileNotFoundError(
            f"Embedding lookup file not found at: {embedding_lookup}"
        )
    with open(embedding_lookup, "r") as f:
        taxonomic_embedding_dict = json.load(f)
    logger.success(
        f"Loaded taxonomic embedding dictionary with {len(taxonomic_embedding_dict)} entries"
    )
    return taxonomic_embedding_dict


def collect_preds(preds, endpoints, device=None):
    """
    Collects the predictions for the specified endpoints.

    Args:
        preds (dict): Dictionary containing predictions for different tasks.
        endpoints (list): List of strings specifying which tasks to collect.

    Returns:
        torch.Tensor: Collected predictions for the specified endpoints.
    """
    if isinstance(endpoints[0], str):
        endpoint_map = {key: idx for idx, key in enumerate(preds.keys())}
        endpoint_indices = torch.tensor(
            [endpoint_map[ep] for ep in endpoints], device=device
        )
    else:
        # if endpoints are already provided as integer indices
        endpoint_indices = endpoints.to(device)

    # Stack preds into [B, 4]
    pred_stack = torch.stack([preds[key] for key in preds.keys()], dim=1)

    # Select predictions by endpoint
    return pred_stack[
        torch.arange(len(preds[next(iter(preds))]), device=device), endpoint_indices
    ]


def endpoint_constraint_loss(preds, wrong_ordering_penalty_weight=0.1, margin=1e-3):
    """
    preds: dict with tasks as keys e.g., ['EC10', 'EC50', 'LOEC', 'NOEC'], each a tensor

    Enforces the biological ordering NOEC <= LOEC <= EC10 <= EC50.
    Each penalty fires when the lower endpoint exceeds the higher one:
      relu(lower + margin - higher)
    which is minimised by pushing lower < higher.
    """
    # Penalty: fire when NOEC > LOEC  (correct: NOEC <= LOEC)
    penalty_noec_loec = (
        F.relu((preds["NOEC"] + margin - preds["LOEC"])).mean()
        if "NOEC" in preds and "LOEC" in preds
        else 0
    )

    # Penalty: fire when EC10 > EC50  (correct: EC10 <= EC50)
    penalty_ec10_ec50 = (
        F.relu((preds["EC10"] + margin - preds["EC50"])).mean()
        if "EC10" in preds and "EC50" in preds
        else 0
    )

    return wrong_ordering_penalty_weight * (penalty_noec_loec + penalty_ec10_ec50)


class TaxonomicRankingLoss(nn.Module):
    """
    Uses the taxonomic tree to retrive common ancestors to the taxids in the forward call and enforces that prediction to be the average of the children predictions.
    The forward call recieves a batch of taxids and predictions, groups them by their parent taxid (e.g., genus, family, etc), predicts the reponse for that embedding, and computes the hierarchical consistency loss.

    The loss is computed as the MSE between the parent taxid prediction and the average of the child taxid predictions.
    The loss is weighted by lambda_hier.

    ## Inputs
    - taxonomic_tree: ete3 Tree object representing the taxonomic hierarchy.
    - lambda_hier: Weight for the hierarchical loss.
    - rank_weights: Weights for sampling taxonomic distance [1, 2, 3, 4, 5, 6, 7 steps].
      Default: [62, 15, 10, 5, 5, 2, 1] (heavily weighted toward genus).
    - detach_species: Whether to detach species predictions to prevent gradient flow.
      We typically want this so loss doesn't affect species predictions.
    - sample_n_inputs: Number of inputs to sample from the batch for computing the loss.
    """

    def __init__(
        self,
        taxonomic_tree: Tree,
        training_taxids: List[str],
        lambda_hier: float = 0.01,
        rank_weights: Dict[str, int] = {
            "genus": 60,
            "family": 15,
            "order": 10,
            "class": 5,
            "subphylum": 4,
            "phylum": 3,
            "kingdom": 2,
            "superkingdom": 1,
        },
        detach_species: bool = True,
        sample_n_inputs: int = 10,
    ):
        super().__init__()
        self.taxonomic_tree = taxonomic_tree
        self.lambda_hier = lambda_hier
        self.rank_weights = rank_weights
        self.training_taxids = set(training_taxids)
        self.detach_species = detach_species
        self.sample_n_inputs = sample_n_inputs

        # Preprocess the tree
        # Remove all leaves not in training taxids
        for leaf in self.taxonomic_tree.iter_leaves():
            if leaf.name not in self.training_taxids:
                _ = leaf.detach()
        # Remove all leaf nodes that are not rank 'species'
        for leaf in self.taxonomic_tree.iter_leaves():
            if leaf.rank != "species":
                _ = leaf.detach()
        # Build leaf node cache for fast lookup
        self.leaf_node_cache = {
            leaf.name: leaf for leaf in self.taxonomic_tree.iter_leaves()
        }
        # Build rank node cache for fast lookup
        self.rank_cache = defaultdict(list)
        for node in self.taxonomic_tree.traverse():
            if getattr(node, "rank", None):
                self.rank_cache[node.rank].append(node)

    def _collect_names_to_keep(self, taxid_list: List[str]) -> set[str]:
        """Return set of node names that are on paths from selected leaves to root."""
        names = set()
        for name in taxid_list:
            node = self.leaf_node_cache.get(name)
            while node:
                if node.name:
                    names.add(node.name)
                node = node.up
        return names

    def prune_tree_to_taxids(self, taxid_list: List[str]) -> set[str]:
        """
        Instead of copying or modifying the tree, return the set of node names
        that belong to the pruned view.
        """
        return self._collect_names_to_keep(taxid_list)

    def leaf_count(self, node, taxid_list):
        return sum(1 for _ in node.iter_leaves() if _.name in taxid_list)

    def sample_ancestor_and_descendants(
        self, taxid_list: List[str]
    ) -> Tuple[str | None, list[str] | None]:
        """
        1. Determine which nodes are relevant for this batch (names_to_keep)
        2. Randomly choose a rank
        3. Among nodes of that rank that are in names_to_keep, find one with
        the largest number of descendant species
        """
        # A set containing names to keep (leaves + ancestors)
        names_to_keep = self.prune_tree_to_taxids(taxid_list)
        if not names_to_keep:
            return None, None

        # Weighted random rank
        chosen_rank = random.choices(
            list(self.rank_weights.keys()), weights=self.rank_weights.values(), k=1
        )[0]

        # Get only nodes at that rank and that are in the pruned set
        possible_ancestors = [
            n for n in self.rank_cache.get(chosen_rank, []) if n.name in names_to_keep
        ]
        if not possible_ancestors:
            return None, None

        # Count descendant leaves
        best_ancestor, best_descendants, best_count = None, [], 0
        for node in possible_ancestors:
            leaves = [leaf for leaf in node.iter_leaves() if leaf.name in taxid_list]
            c = len(leaves)
            if c > best_count:
                best_ancestor, best_descendants, best_count = node, leaves, c

        if not best_ancestor:
            return None, None

        return best_ancestor.name, [leaf.name for leaf in best_descendants]

    def subsample_batch(self, inputs):
        """
        Subsample the batch inputs.

        Args:
            inputs: Original batch of inputs.
            sample_n_inputs: Number of inputs to sample.

        Returns:
            Subsampled inputs and the actual number of samples obtained.
        """
        # Get the actual batch size
        actual_batch_size = list(inputs.values())[0].shape[0]

        # Take minimum of requested samples and actual batch size
        actual_n_inputs = min(self.sample_n_inputs, actual_batch_size)

        # Sample random inputs, this is a dictionary holding various tensors
        # We simply sample the first n from the batch dimension since the batch is already randomized during training
        random_inputs = {k: v[:actual_n_inputs, :] for k, v in inputs.items()}

        return random_inputs, actual_n_inputs

    def forward(self, model, inputs, taxids):
        """
        Compute hierarchical consistency loss to enforce ancestor predictions = average of descendant species predictions.

        Uses intelligent sampling to select an ancestor at a meaningful taxonomic level (genus, family, etc.)
        rather than randomly ending up at root/kingdom level.

        Args:
            model: The model (for getting ancestor predictions).
            inputs: a batch of inputs to the model.
            taxids: List of taxid strings corresponding to the inputs.

        Strategy:
        1. Prune tree to only include taxids from batch
        2. Weighted random sample of taxonomic distance (heavily weighted toward genus)
        3. Find the ancestor at that distance with the most species descendants
        4. Build batch with those species and compute loss

        The forward pass builds a new small batch on the fly looking like this:
        Batch Index  | Input  | Taxid
        -------------|--------|--------
        0-(N-1)      | Input0 | Species_1, Species_2, ..., Species_N
        N-(2N-1)     | Input1 | Species_1, Species_2, ..., Species_N
        ...          | ...    | ...

        Where N is the number of descendant species found for the chosen ancestor.

        Returns:
            Weighted hierarchical loss (scalar tensor)
        """
        device = list(inputs.values())[0].device

        sampled_inputs, actual_n_inputs = self.subsample_batch(inputs)

        # Get unique taxids from the batch
        unique_taxids = list(set(taxids))

        # Intelligently sample an ancestor and its descendants using pruned tree
        ancestor_taxid, descendant_taxids = self.sample_ancestor_and_descendants(
            unique_taxids,
        )

        if (
            ancestor_taxid is None
            or descendant_taxids is None
            or len(descendant_taxids) < 2
        ):
            return torch.tensor(0.0, device=device)

        sample_n_taxids = len(descendant_taxids)

        # Build a new minibatch structured as: [input1_species1, input1_species2, ..., input1_speciesN, input2_species1, ...]
        extended_inputs = {}
        for k, v in sampled_inputs.items():
            # Repeat each input sample_n_taxids times consecutively
            # Shape: [actual_n_inputs * sample_n_taxids, ...]
            extended_inputs[k] = v.repeat_interleave(sample_n_taxids, dim=0)

        # Create taxid list that cycles through all descendant taxids for each input
        # Use actual_n_inputs instead of sample_n_inputs to handle small batches correctly
        extended_taxids = list(descendant_taxids) * actual_n_inputs

        # Get predictions from the model
        if self.detach_species:
            with torch.no_grad():
                preds, _ = model(**extended_inputs, taxids=extended_taxids)
        else:
            preds, _ = model(**extended_inputs, taxids=extended_taxids)

        # Combine preds (output is a dictionary) to a matrix
        preds = torch.stack([preds[key] for key in preds.keys()], dim=1)

        # Reshape predictions to [actual_n_inputs, sample_n_taxids, n_endpoints]
        preds_reshaped = preds.view(actual_n_inputs, sample_n_taxids, -1)

        # Calculate average prediction per unique input across all child taxids
        # Shape: [actual_n_inputs, n_endpoints]
        avg_child_preds = preds_reshaped.mean(dim=1)

        # Get predictions for the ancestor taxid with each input
        ancestor_inputs = sampled_inputs  # Already has actual_n_inputs samples
        ancestor_taxids = [ancestor_taxid] * actual_n_inputs

        # Forward pass for ancestor
        ancestor_preds, _ = model(**ancestor_inputs, taxids=ancestor_taxids)
        ancestor_preds = torch.stack(
            [ancestor_preds[key] for key in ancestor_preds.keys()], dim=1
        )
        # Shape: [actual_n_inputs, n_endpoints]

        # Compute the MSE loss between ancestor predictions and average child predictions
        hierarchical_loss = F.mse_loss(ancestor_preds, avg_child_preds)

        # Apply lambda weight
        hierarchical_loss = self.lambda_hier * hierarchical_loss

        return hierarchical_loss


class RegressionModule(nn.Module):
    """
    DNN module for Regression.

    A flexible, dynamic feed-forward neural network that can act as a standalone regressor
    or be integrated as a regression head for other models like transformers.

    ## Inputs
    - input_layer_size: Number of neurons in input layer.
    - hidden_layer_sizes: A list specifying the size of each hidden layer. If None, no hidden layers are used.
    - dropout: Dropout rate for regularization.
    - activation: Activation function ('GELU', 'ReLU', 'LeakyReLU', etc.).
    - n_outputs: Number of output neurons. 1 is standard regression. >1 is MultiTaskRegression.
    """

    def __init__(
        self,
        input_layer_size: int,
        hidden_layer_sizes: Union[List[int], None],
        dropout: float,
        activation: str = "GELU",
        n_outputs: int = 1,
    ):
        super().__init__()
        self.input_layer_size = input_layer_size
        self.hidden_layer_sizes = (
            hidden_layer_sizes or []
        )  # Default to an empty list if None

        # Choose activation function dynamically
        self.activation_fn = getattr(nn, activation, nn.GELU)()

        # Dynamically create hidden layers
        self.hidden_layers = nn.ModuleList()
        for size in self.hidden_layer_sizes:
            self.hidden_layers.append(nn.Linear(input_layer_size, size))
            input_layer_size = size  # Update input size for the next layer

        # Output layer for logits
        self.out_proj = nn.Linear(input_layer_size, n_outputs)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        """
        Forward pass through the DNN.
        """
        x = inputs

        # Apply each hidden layer sequentially
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation_fn(x)
            x = self.dropout(x)

        # Output layer
        x = self.out_proj(x).squeeze(1)

        return x


class MultiTaskRegressionModule(nn.Module):
    """
    MultiTask DNN module for Regression.

    This class combines multiple RegressionModule models to create a multitask neural network for regression.
    Each task can have a unique architecture with different numbers of layers, neurons, and outputs.

    ## Inputs
    - input_sizes: A dictionary specifying input sizes for each task (key: task name, value: input size).
    - task_configs: A dictionary specifying configurations for each task (key: task name, value: dict with keys:
        - 'hidden_layer_sizes': List[int], sizes of hidden layers.
        - 'dropout': float, dropout rate.
        - 'activation': str, activation function.
        - 'n_outputs': int, number of output neurons for the task).
    - fusion: Whether to perform data fusion by combining outputs of tasks for cross-task extrapolation.
    """

    def __init__(
        self,
        input_layer_size: int,
        regression_task_configs: Dict[str, Dict],
        fusion_network_config: Union[Dict[str, Dict], None] = None,
    ):
        super().__init__()

        self.tasks = nn.ModuleDict()
        self.task_names = list(regression_task_configs.keys())
        self.fusion = False
        self.input_layer_size = input_layer_size

        if fusion_network_config is not None:
            # This means we are building some common layers for both tasks
            self.fusion_layers = RegressionModule(
                input_layer_size=self.input_layer_size,
                hidden_layer_sizes=fusion_network_config["hidden_layer_sizes"],
                dropout=fusion_network_config["dropout"],
                activation=fusion_network_config.get("activation", "ReLU"),
                n_outputs=fusion_network_config["n_outputs"],
            )
            self.fusion = True

        # Create individual task-specific regressors
        for task_name, config in regression_task_configs.items():
            if fusion_network_config is not None:
                input_layer_size = fusion_network_config["n_outputs"]
            self.tasks[task_name] = RegressionModule(
                input_layer_size=input_layer_size,
                hidden_layer_sizes=config["hidden_layer_sizes"],
                dropout=config["dropout"],
                activation=config.get("activation", "ReLU"),
                n_outputs=config["n_outputs"],
            )

    def forward(self, inputs):
        """
        Forward pass through the multitask network.

        ## Outputs
        - task_outputs: A dictionary with task names as keys and their respective outputs as values.
        """
        x = inputs
        if self.fusion:
            x = self.fusion_layers(x)

        task_outputs = {}

        # Forward pass through each task-specific model
        for task_name, task_layers in self.tasks.items():
            task_outputs[task_name] = task_layers(x)

        return task_outputs


class TaxonomicEmbedder(nn.Module):
    """
    Taxonomic Embedding module that handles lookup and projection of taxonomic embeddings.

    Args:
        taxonomic_embedding_dict (Dict[str, torch.Tensor or np.ndarray]):
            Dictionary mapping taxid strings to embedding vectors (typically 768-dim).
        embedding_dim (int):
            Dimension of the taxonomic embeddings (default: 768).
        dropout (float):
            Dropout rate applied after projection and normalization (default: 0.1).

    Input:
        taxids (List[str] or List[List[str]]):
            Batch of taxid strings. Can be single taxids per sample or multiple taxids per sample.

    Output:
        torch.Tensor: Projected and normalized taxonomic embeddings of shape [batch_size, 1, embedding_dim]
                      or [batch_size, n_taxids, embedding_dim] if multiple taxids per sample.

    Example:
        >>> embedder = TaxonomicEmbedder(taxid_dict, embedding_dim=768, dropout=0.1)
        >>> taxids = ["9606", "10090", "7227"]  # Human, Mouse, Fly
        >>> embeddings = embedder(taxids)  # Shape: [3, 1, 768]
    """

    def __init__(
        self,
        taxonomic_embedding_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
        embedding_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        if not isinstance(taxonomic_embedding_dict, dict):
            raise ValueError(
                "taxonomic_embedding_dict must be a dictionary mapping taxid strings to embedding vectors."
            )
        self.taxonomic_embedding_dict = taxonomic_embedding_dict

        # Learnable projection to align external embeddings with transformer space
        self.external_proj = nn.Linear(embedding_dim, embedding_dim)

        # Normalization after projection
        self.external_norm = nn.LayerNorm(embedding_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize projection weights (small init for stability)
        nn.init.xavier_uniform_(self.external_proj.weight)
        nn.init.zeros_(self.external_proj.bias)

    def lookup_embeddings(
        self, taxids: Union[List[str], List[List[str]]]
    ) -> torch.Tensor:
        """
        Look up embeddings from the taxonomic dictionary.

        Args:
            taxids: List of taxid strings or list of lists of taxid strings (can contain None values)

        Returns:
            torch.Tensor: Raw embeddings from dictionary [batch_size, n_taxids, embedding_dim]
        """
        # Handle single taxid per sample (or None)
        if taxids[0] is None or isinstance(taxids[0], str):
            embeddings = []
            for taxid in taxids:
                # Use zero embedding for None taxids
                if taxid is None:
                    emb = np.zeros(self.embedding_dim, dtype=np.float32)
                else:
                    emb = self.taxonomic_embedding_dict.get(
                        taxid.split("_")[0]
                        .split(",")[0]
                        .split(".")[0]
                        .strip(),  # We strip any suffixes after underscore, comma and period, we only encode actual taxids
                        np.zeros(self.embedding_dim, dtype=np.float32),
                    )
                # Convert to numpy if needed
                if isinstance(emb, list):
                    emb = np.array(emb, dtype=np.float32)
                embeddings.append(emb)

            embeddings = np.stack(embeddings)  # [batch_size, embedding_dim]
            # Sum all embeddings in the batch to check if the sum is zero (indicating all taxids were None or missing)
            if np.sum(embeddings) == 0:
                # If all embeddings are zero, we can skip the projection and return zeros directly
                logger.warning(
                    "All taxids in the batch are None or missing from the embedding dictionary. Consider handling this case separately to avoid unnecessary computation."
                )
            embeddings = torch.from_numpy(embeddings).float()
            embeddings = embeddings.unsqueeze(1)  # [batch_size, 1, embedding_dim]

        # Handle multiple taxids per sample
        else:
            embeddings = []
            for taxid_list in taxids:
                sample_embeddings = []
                for taxid in taxid_list:
                    emb = self.taxonomic_embedding_dict.get(
                        taxid.split("_")[0]
                        .split(",")[0]
                        .split(".")[0]
                        .strip(),  # We strip any suffixes after underscore, comma and period, we only encode actual taxids
                        np.zeros(self.embedding_dim, dtype=np.float32),
                    )
                    if isinstance(emb, list):
                        emb = np.array(emb, dtype=np.float32)
                    sample_embeddings.append(emb)
                embeddings.append(np.stack(sample_embeddings))

            embeddings = np.stack(embeddings)  # [batch_size, n_taxids, embedding_dim]
            embeddings = torch.from_numpy(embeddings).float()

        return embeddings

    def forward(self, taxids: Union[List[str], List[List[str]]]) -> torch.Tensor:
        """
        Forward pass: lookup, project, normalize, and apply dropout.

        Args:
            taxids: Batch of taxid strings

        Returns:
            torch.Tensor: Processed embeddings [batch_size, n_taxids, embedding_dim]
        """
        # Lookup raw embeddings from dictionary
        embeddings = self.lookup_embeddings(taxids)

        # Move to same device as projection layer
        embeddings = embeddings.to(self.external_proj.weight.device)

        # Project to transformer space
        embeddings = self.external_proj(embeddings)
        embeddings = self.external_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MultimodalRoBERTa(nn.Module):
    """
    Multimodal RoBERTa module for integrating external embeddings (e.g., taxonomic embeddings)
    as pseudo-tokens into a RoBERTa-based model.
    """

    def __init__(
        self,
        base_model,
        padding_idx: int = 1,
        prepend_external: bool = True,
        taxonomic_embedding_dict: Dict[str, Union[torch.Tensor, np.ndarray]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.base_model_type = base_model.config.model_type
        self.embedding_dim = base_model.embeddings.word_embeddings.embedding_dim
        self.padding_idx = padding_idx
        self.prepend_external = prepend_external
        self.taxonomic_embedding_dict = taxonomic_embedding_dict
        if self.taxonomic_embedding_dict is not None:
            self.taxonomic_embedder = TaxonomicEmbedder(
                taxonomic_embedding_dict=self.taxonomic_embedding_dict,
                embedding_dim=self.embedding_dim,
                dropout=0.1,
            )

    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        """Replace non-padding symbols with their position numbers. Position numbers begin at
        padding_idx+1. Padding symbols are ignored."""
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
        return incremental_indices.long() + padding_idx

    def combine_embeddings(
        self,
        input_embeddings,
        input_attention_mask,
        input_position_ids,
        external_embeds,
    ):
        """
        Combine SMILES input embeddings with projected taxonomic embedding.
        The external embedding is treated as a single pseudo-token.
        """
        if external_embeds.dim() == 2:
            external_embeds = external_embeds.unsqueeze(1)  # [batch, 1, 768]
        elif (
            external_embeds.dim() != 3 or external_embeds.size(-1) != self.embedding_dim
        ):
            raise ValueError(
                f"Expected external_embeds shape [batch, 768] or [batch, n_ext, 768], got {external_embeds.shape}"
            )

        # Build external token mask (always active)
        external_mask = torch.ones(
            (input_attention_mask.size(0), 1),
            dtype=input_attention_mask.dtype,
            device=input_attention_mask.device,
        )

        # Position 0 is never used by RoBERTa under normal operation (real tokens
        # start at padding_idx+1, padding uses padding_idx). Assigning 0 here gives
        # the taxonomic token its own dedicated, learnable position embedding without
        # colliding with padding tokens, while still being position-agnostic in intent.
        external_position_ids = torch.zeros(
            (input_position_ids.size(0), 1),
            dtype=input_position_ids.dtype,
            device=input_position_ids.device,
        )

        # Concatenate at front, but after CLS token (position 0)
        if self.prepend_external:
            combined_input_embeddings = torch.cat(
                [
                    input_embeddings[:, :1, :],
                    external_embeds,
                    input_embeddings[:, 1:, :],
                ],
                dim=1,
            )
            combined_attention_mask = torch.cat(
                [
                    input_attention_mask[:, :1],
                    external_mask,
                    input_attention_mask[:, 1:],
                ],
                dim=1,
            )
            combined_position_ids = torch.cat(
                [
                    input_position_ids[:, :1],
                    external_position_ids,
                    input_position_ids[:, 1:],
                ],
                dim=1,
            )
        else:
            combined_input_embeddings = torch.cat(
                [input_embeddings, external_embeds], dim=1
            )
            combined_attention_mask = torch.cat(
                [input_attention_mask, external_mask], dim=1
            )
            combined_position_ids = torch.cat(
                [input_position_ids, external_position_ids], dim=1
            )

        return combined_input_embeddings, combined_attention_mask, combined_position_ids

    def forward(self, input_ids, attention_mask, taxids=None, verbose=False):
        # Obtain token embeddings from the model embedding layer
        input_embeddings = self.base_model.embeddings.word_embeddings(input_ids)

        # Generate position IDs for tokens
        position_ids = self.create_position_ids_from_input_ids(
            input_ids=input_ids, padding_idx=self.padding_idx
        )

        # Obtain external embeddings from taxonomic embedder if taxids provided
        if taxids is not None and self.taxonomic_embedding_dict is not None:
            external_embeds = self.taxonomic_embedder(taxids=taxids)
            # Combine if with input embeddings
            input_embeddings, attention_mask, position_ids = self.combine_embeddings(
                input_embeddings=input_embeddings,
                input_attention_mask=attention_mask,
                input_position_ids=position_ids,
                external_embeds=external_embeds,
            )

        if verbose:
            logger.info(f"input_embeddings shape: {input_embeddings.shape}")
            logger.info(f"attention_mask shape: {attention_mask.shape}")
            logger.info(f"position_ids shape: {position_ids.shape}")

        # Forward through base model
        outputs = self.base_model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        return outputs


class TRIDENT2(nn.Module):
    """
    # TRIDENT2
    Class to build Transformer and DNN structure.

    ## Inputs
    - transformer_encoder: a pytorch transformer (BERT-like) model
    - dnn: an instance of the DNN_module class
    """

    def __init__(self, transformer_encoder, dnn):
        super().__init__()
        self.transformer_encoder = transformer_encoder
        self.dnn = dnn

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        duration=None,
        one_hot_encoding=None,
        taxids=None,
        featurized_inputs=None,
        verbose=False,
    ):
        if self.transformer_encoder.base_model_type == "roberta":
            transformer_encoder_output = self.transformer_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                taxids=taxids,
                verbose=verbose,
            )[0][:, 0, :]  # Last hidden state NOT pooler output, then CLS
        elif self.transformer_encoder.base_model_type == "molformer":
            transformer_encoder_output = self.transformer_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                taxids=taxids,
                verbose=verbose,
            ).pooler_output  # Molformer is trained with pooler output, not CLS
        elif self.transformer_encoder.base_model_type == "MAT":
            transformer_encoder_output = self.transformer_encoder(
                featurized_inputs=featurized_inputs, taxids=taxids, input_ids=input_ids
            ).mean(axis=1)  # Molformer is trained with pooler output, not CLS
        else:
            logger.warning(
                "Model type not supported, defaulting to RoBERTa (using CLS token as output)."
            )
            transformer_encoder_output = self.transformer_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                taxids=taxids,
                verbose=verbose,
            )[0][:, 0, :]  # Last hidden state NOT pooler output, then CLS

        inputs = torch.cat(
            [
                x
                for x in [
                    transformer_encoder_output,
                    duration.view(input_ids.size(0), 1),
                    one_hot_encoding,
                ]
                if x is not None
            ],
            axis=1,
        )
        out = self.dnn(inputs)

        if verbose:
            logger.info("Inputs to DNN:\n", inputs)
            logger.info("Inputs to DNN shape:\n", inputs.shape)

        return out, transformer_encoder_output


class MultiModalMAT(nn.Module):
    def __init__(
        self,
        MATmodel,
        MAT_model_dim,
        external_embed_dim,
        num_layers=1,
        num_heads=8,
        hidden_dim=768,
        dim_feedforward=3072,
        vocab_size=10,
    ):
        super(MultiModalMAT, self).__init__()
        self.MATmodel = MATmodel
        self.MAT_model_dim = MAT_model_dim
        self.external_embed_dim = external_embed_dim
        self.base_model_type = "MAT"

        # Ensure both embeddings are projected to the same size
        self.MAT_model_dim_projection = nn.Linear(MAT_model_dim, hidden_dim)
        self.external_embed_dim_projection = nn.Linear(external_embed_dim, hidden_dim)
        self.word_embeds = nn.Embedding(vocab_size, hidden_dim)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, featurized_inputs, external_embeds, input_ids=None):
        # Project both modalities to the same hidden dimension
        if input_ids is not None:
            word_embeddings = self.word_embeds(input_ids).unsqueeze(
                1
            )  # (batch_size, 1, hidden_dim)
        batch_mask = torch.sum(torch.abs(featurized_inputs.node_features), dim=-1) != 0
        embedded = self.MATmodel.src_embed(featurized_inputs.node_features)
        encoder_output = self.MATmodel.encoder(
            embedded,
            batch_mask,
            adj_matrix=featurized_inputs.adjacency_matrix,
            distance_matrix=featurized_inputs.distance_matrix,
        )
        MAT_out = self.MAT_model_dim_projection(
            encoder_output
        )  # (batch_size, smiles_seq_len, hidden_dim)
        external_embeds = self.external_embed_dim_projection(
            external_embeds
        )  # (batch_size, word_seq_len, hidden_dim)

        # Concatenate both modalities along the sequence dimension
        if input_ids is not None:
            combined_input = torch.cat(
                [MAT_out, external_embeds, word_embeddings], dim=1
            )  # (batch_size, total_seq_len, hidden_dim)
        else:
            combined_input = torch.cat([MAT_out, external_embeds], dim=1)

        # Pass through transformer encoder
        encoded_output = self.encoder(combined_input)

        return encoded_output  # (batch_size, total_seq_len, hidden_dim)
