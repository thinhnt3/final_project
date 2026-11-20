"""AdaMF-MAT image encoder: adaptive fusion of an image's modalities, trained against a generator.

Follows "Unleashing the Power of Imbalanced Modality Information for Multi-modal Knowledge
Graph Completion" (Zhang et al., COLING 2024) and its code (github.com/zjukg/AdaMF-MAT),
adapted to text-to-image retrieval: an image is the head entity of the single relation
"is described by" and the caption is the tail.

- AdaMF: every modality is projected to one size; a weight per modality comes from a shared
  vector applied to tanh of the embedding, softmax over the modalities; the joint embedding
  is the weighted sum.
- MAT: a generator makes synthetic image and text embeddings from the structure embedding
  and noise. The encoder (discriminator) must score the real entity above the synthetic
  one; the generator is then updated to close that gap.

Differences from the paper, all forced by the retrieval setting:
- structure embedding: test images are unseen, so it cannot be a per-entity vector learned
  from scratch; it is the state of the image node after GraphSAGE over the scene graph;
- decoder: the joint embedding is projected to the CLIP space and scored by cosine with the
  frozen CLIP caption vector, trained with in-batch InfoNCE, as every graph method of this
  repository, instead of RotatE with self-adversarial negative sampling;
- adversarial term: the tail is a caption and has no modalities, so only the head can be
  synthetic (the paper also replaces the tail, and both). Per pair, the real and the
  synthetic head compete for the caption in a two-way softmax, which is the paper's
  "real scores above synthetic" on the scale of the InfoNCE logits.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from src.core.exceptions import ConfigError
from src.models.graph_data import NodeKind
from src.models.graphsage import GraphImageEncoder, GraphSageConfig, ImageNodeFeature
from src.models.train import info_nce

# Order of the modalities in stacked tensors and in `modality_weights`.
MODALITIES = ("structure", "image", "text")


def label_text_vectors(concept_vectors: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Mean CLIP text vector of each image's concept labels, shape (graphs, clip dim); zero without objects.

    Labels are the bare concept words (no prompt template); a concept appears once per image.
    """
    is_object = batch.node_kind == NodeKind.OBJECT
    return scatter(
        concept_vectors[batch.feature_row[is_object]], batch.batch[is_object],
        dim=0, dim_size=batch.num_graphs, reduce="mean",
    )


@dataclass(frozen=True)
class AdamfMatConfig:
    """Architecture and adversarial settings; the switches are the paper's ablations."""

    dim: int
    adaptive_fusion: bool  # False: the joint embedding is the mean of the modalities ("w/o AdaMF")
    mat: bool              # False: no generator, plain training ("w/o MAT")
    mu: float              # weight of the adversarial term in the encoder's loss
    noise_dim: int
    generator_hidden_dim: int
    generator_learning_rate: float
    # Modalities fused, in the order of `MODALITIES`; dropping one is a modality ablation.
    modalities: tuple[str, ...] = MODALITIES
    # True: each modality embedding is L2-normalized before fusion, so the fusion weights are
    # not offset by the norm of the GraphSAGE output (the paper's structure embedding is bounded).
    normalize_modalities: bool = False
    # Training only: share of images whose image and text modalities are hidden from the
    # fusion, so the structure modality alone must match the caption. 0 switches it off.
    structure_only_rate: float = 0.0
    # Weight of an extra InfoNCE term on the structure modality alone, for every image, so
    # the graph branch always has to match the caption by itself. 0 switches it off.
    structure_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        if min(self.dim, self.noise_dim, self.generator_hidden_dim) < 1:
            raise ConfigError("adamf_mat.dim, noise_dim and generator_hidden_dim must be >= 1")
        if self.mu <= 0 or self.generator_learning_rate <= 0:
            raise ConfigError("adamf_mat.mu and generator_learning_rate must be > 0; switch MAT off with mat: false")
        unknown = set(self.modalities) - set(MODALITIES)
        if not self.modalities or unknown or len(set(self.modalities)) != len(self.modalities):
            raise ConfigError(f"adamf_mat.modalities must be distinct values of {MODALITIES}, got {self.modalities}")
        if tuple(m for m in MODALITIES if m in self.modalities) != tuple(self.modalities):
            raise ConfigError(f"adamf_mat.modalities must follow the order {MODALITIES}")
        # The generator makes the image and text modalities from the structure one.
        if self.mat and tuple(self.modalities) != MODALITIES:
            raise ConfigError("adamf_mat.mat needs all three modalities; drop a modality with mat: false")
        if not 0.0 <= self.structure_only_rate < 1.0:
            raise ConfigError(f"adamf_mat.structure_only_rate must lie in [0, 1), got {self.structure_only_rate}")
        if self.structure_loss_weight < 0 or (self.structure_loss_weight > 0 and "structure" not in self.modalities):
            raise ConfigError("adamf_mat.structure_loss_weight must be >= 0 and needs the structure modality")
        alone = "structure" not in self.modalities or len(self.modalities) < 2
        if self.structure_only_rate > 0 and (self.mat or alone):
            raise ConfigError("adamf_mat.structure_only_rate needs mat: false, the structure modality and another one")

    def run_name(self) -> str:
        """Name of training runs, e.g. `adamf-mat-d256-mu1`, `adamf-d256-l2` or `mean-d256-no-text`."""
        fusion = "adamf" if self.adaptive_fusion else "mean"
        name = f"{fusion}-mat-d{self.dim}-mu{self.mu:g}" if self.mat else f"{fusion}-d{self.dim}"
        if self.normalize_modalities:
            name += "-l2"
        dropped = [m for m in MODALITIES if m not in self.modalities]
        name += "".join(f"-no-{m}" for m in dropped)
        name += f"-so{self.structure_only_rate:g}" if self.structure_only_rate > 0 else ""
        return name + (f"-sl{self.structure_loss_weight:g}" if self.structure_loss_weight > 0 else "")


class ModalityGenerator(nn.Module):
    """Structure embedding and noise → a synthetic embedding of one modality."""

    def __init__(self, config: AdamfMatConfig) -> None:
        super().__init__()
        self.noise_dim = config.noise_dim
        self.layers = nn.Sequential(
            nn.Linear(config.noise_dim + config.dim, config.generator_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.generator_hidden_dim, config.dim),
        )

    def forward(self, structure: torch.Tensor) -> torch.Tensor:
        """One synthetic embedding per row, shape (batch, dim); a new noise draw on every call."""
        noise = torch.randn(structure.shape[0], self.noise_dim, device=structure.device)
        result: torch.Tensor = self.layers(torch.cat([noise, structure], dim=-1))
        return result


class AdamfMatImageEncoder(nn.Module):
    """Sub-graph of an image → unit vector comparable with CLIP text vectors, through AdaMF."""

    concept_vectors: torch.Tensor
    image_vectors: torch.Tensor

    def __init__(
        self,
        config: AdamfMatConfig,
        structure_config: GraphSageConfig,
        concept_vectors: torch.Tensor,
        predicate_vectors: torch.Tensor,
        image_vectors: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.adversarial = config.mat
        self.structure_loss_weight = config.structure_loss_weight
        self.generator_learning_rate = config.generator_learning_rate
        clip_dim = concept_vectors.shape[1]
        # Frozen lookup tables; restored from the embedding store, not saved in checkpoints.
        self.register_buffer("concept_vectors", concept_vectors.float(), persistent=False)
        self.register_buffer("image_vectors", image_vectors.float(), persistent=False)
        uses = set(config.modalities)
        if "structure" in uses:
            # The image node carries no CLIP image vector here: that is the image modality's job.
            graph_only = replace(structure_config, image_node=ImageNodeFeature.LEARNED)
            self.structure = GraphImageEncoder(graph_only, concept_vectors, predicate_vectors)
            self.structure_proj = nn.Linear(structure_config.hidden_dim, config.dim)
        if "image" in uses:
            self.image_proj = nn.Linear(clip_dim, config.dim)
        if "text" in uses:
            self.text_proj = nn.Linear(clip_dim, config.dim)
        self.attention = nn.Linear(config.dim, 1, bias=False)
        self.project = nn.Linear(config.dim, clip_dim)
        if config.mat:
            self.image_generator = ModalityGenerator(config)
            self.text_generator = ModalityGenerator(config)

    def _scaled(self, embedding: torch.Tensor) -> torch.Tensor:
        """The embedding itself, or its unit vector when modalities are normalized before fusion."""
        return nn.functional.normalize(embedding, dim=-1) if self.config.normalize_modalities else embedding

    def modality_embeddings(self, batch: Batch) -> tuple[torch.Tensor, ...]:
        """Embedding of each used modality, in the order of `MODALITIES`, each of shape (graphs, dim)."""
        kind, row = batch.node_kind, batch.feature_row
        embeddings = []
        for modality in self.config.modalities:
            if modality == "structure":
                embedding = self.structure_proj(self.structure.image_node_state(batch))
            elif modality == "image":
                embedding = self.image_proj(self.image_vectors[row[kind == NodeKind.IMAGE]])
            else:
                embedding = self.text_proj(label_text_vectors(self.concept_vectors, batch))
            embeddings.append(self._scaled(embedding))
        return tuple(embeddings)

    def modality_weights(self, *embeddings: torch.Tensor, present: torch.Tensor | None = None) -> torch.Tensor:
        """Weight of each modality per image, shape (batch, modalities), in the order given; rows sum to 1.

        This is the adaptive fusion of AdaMF: u_m = aᵀ·tanh(e_m), w = softmax(u). Weights are computed
        per image, so the mix can differ from one image to the next.
        `present` (batch, modalities, bool) hides modalities, as structure-only dropout does: a hidden
        modality gets weight 0 and the visible ones share the whole weight.
        """
        stacked = torch.stack(embeddings, dim=1)  # (batch, modalities, dim)
        if present is None:
            present = torch.ones(stacked.shape[:2], dtype=torch.bool, device=stacked.device)
        if not self.config.adaptive_fusion:
            # "w/o AdaMF" ablation: plain mean over the visible modalities.
            return present.float() / present.sum(dim=1, keepdim=True)
        # One score per modality from a shared vector `a` (Linear dim→1, no bias). tanh bounds each
        # component, so a long embedding cannot raise its own score; without L2 normalization it still
        # dominates the fused sum through its length, which these weights alone do not show.
        logits = self.attention(torch.tanh(stacked)).squeeze(-1)  # (batch, modalities)
        # -inf before softmax gives hidden modalities exactly 0 and renormalizes over the rest.
        return torch.softmax(logits.masked_fill(~present, float("-inf")), dim=-1)

    def structure_only_mask(self, rows: int, device: torch.device) -> torch.Tensor:
        """Modalities present per row: every one, except on a random `structure_only_rate` of rows.

        On those rows only the structure modality stays visible, so the fused vector, and the loss,
        depend on the scene graph alone; the image and text shortcuts cannot answer the caption.
        """
        present = torch.ones(rows, len(self.config.modalities), dtype=torch.bool, device=device)
        # A fresh Bernoulli draw per row and per batch: about `structure_only_rate` of rows are picked.
        hidden = torch.rand(rows, device=device) < self.config.structure_only_rate
        others = [i for i, m in enumerate(self.config.modalities) if m != "structure"]
        # Picked rows × non-structure columns are set to False (hidden).
        present[hidden.nonzero(as_tuple=True)[0].unsqueeze(1), torch.tensor(others, device=device)] = False
        return present

    def fuse(self, *embeddings: torch.Tensor, present: torch.Tensor | None = None) -> torch.Tensor:
        """Unit vector in the CLIP space of the joint embedding, shape (batch, clip dim)."""
        weights = self.modality_weights(*embeddings, present=present).unsqueeze(-1)
        joint = (weights * torch.stack(embeddings, dim=1)).sum(dim=1)
        return nn.functional.normalize(self.project(joint), dim=-1)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Unit-length vector of each image in the batch, shape (graphs, clip dim)."""
        embeddings = self.modality_embeddings(batch)
        if self.training and self.config.structure_only_rate > 0:
            present = self.structure_only_mask(embeddings[0].shape[0], embeddings[0].device)
            return self.fuse(*embeddings, present=present)
        return self.fuse(*embeddings)

    def fused_vectors(self, batch: Batch) -> torch.Tensor:
        """Joint vector with every modality, also in training mode (no structure-only rows)."""
        return self.fuse(*self.modality_embeddings(batch))

    def structure_vectors(self, batch: Batch) -> torch.Tensor:
        """Unit vector in the CLIP space from the structure modality alone."""
        embeddings = self.modality_embeddings(batch)
        present = torch.zeros(embeddings[0].shape[0], len(embeddings), dtype=torch.bool, device=embeddings[0].device)
        present[:, self.config.modalities.index("structure")] = True
        return self.fuse(*embeddings, present=present)

    def generator_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters of the two modality generators; none when MAT is off."""
        if self.config.mat:
            yield from self.image_generator.parameters()
            yield from self.text_generator.parameters()

    def _synthetic(self, structure: torch.Tensor) -> torch.Tensor:
        """Vector of each entity with its image and text embeddings replaced by generated ones."""
        image, text = self.image_generator(structure), self.text_generator(structure)
        return self.fuse(structure, self._scaled(image), self._scaled(text))

    def discriminator_loss(self, batch: Batch, texts: torch.Tensor, temperature: float) -> torch.Tensor:
        """InfoNCE on the real entities, plus mu times the loss of losing to the synthetic ones."""
        structure, image, text = self.modality_embeddings(batch)
        real = self.fuse(structure, image, text)
        gap = ((self._synthetic(structure) - real) * texts).sum(dim=-1) / temperature
        return info_nce(real, texts, temperature) + self.config.mu * nn.functional.softplus(gap).mean()

    def generator_loss(self, batch: Batch, texts: torch.Tensor, temperature: float) -> torch.Tensor:
        """Loss of the synthetic entities losing to the real ones, with the encoder held fixed."""
        with torch.no_grad():
            structure, image, text = self.modality_embeddings(batch)
            real = self.fuse(structure, image, text)
        gap = ((real - self._synthetic(structure)) * texts).sum(dim=-1) / temperature
        return nn.functional.softplus(gap).mean()
