"""AdaMF-MAT on a knowledge graph: adaptive modality fusion, RotatE scoring, adversarial training.

Follows "Unleashing the Power of Imbalanced Modality Information for Multi-modal Knowledge
Graph Completion" (Zhang et al., COLING 2024) and its code (github.com/zjukg/AdaMF-MAT,
`AdvMixRotatE`, `AdvMixTrainer`, `MultiGenerator`).

- AdaMF: an entity has a learned structure embedding, an image embedding and a text
  embedding, all of one size. A shared vector scores tanh of each; softmax over the modalities
  gives the weights of the joint embedding.
- RotatE: a relation rotates the head's joint embedding; the score is the margin minus the
  distance to the tail's.
- MAT: generators make image and text embeddings from the structure embedding and noise. The
  encoder must score a true triple above the same triple with a synthetic head, tail or both;
  the generators are then updated to close that gap.

Differences from the released code:
- image and text inputs are frozen CLIP vectors (the paper uses BEiT and sentence-BERT
  features and lets them be fine-tuned);
- negatives are uniform corruptions of the head or the tail, alternating by batch, without
  the Bernoulli trick or the removal of accidental true triples.
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn

from src.core.exceptions import ConfigError, DataError

# Order of the modalities in stacked tensors and in `modality_weights`.
MODALITIES = ("structure", "image", "text")


@dataclass(frozen=True)
class KgAdamfMatConfig:
    """Architecture, loss and adversarial settings; the switches are the paper's ablations."""

    dim: int                # size of a relation phase vector; entity embeddings are twice that
    margin: float
    epsilon: float          # with the margin, sets the range of the initial embeddings
    negatives: int          # corrupted triples per true triple
    adversarial_temperature: float  # sharpness of the self-adversarial weights of the negatives
    adaptive_fusion: bool   # False: the joint embedding is the mean of the modalities
    mat: bool               # False: no generator, plain training
    mu: float               # weight of the adversarial terms in the encoder's loss
    noise_dim: int
    generator_hidden_dim: int
    generator_learning_rate: float
    # Modalities fused, in the order of `MODALITIES`; ("structure",) is plain RotatE.
    modalities: tuple[str, ...] = MODALITIES

    def __post_init__(self) -> None:
        if min(self.dim, self.negatives, self.noise_dim, self.generator_hidden_dim) < 1:
            raise ConfigError("adamf_mat.dim, negatives, noise_dim and generator_hidden_dim must be >= 1")
        if min(self.margin, self.epsilon, self.adversarial_temperature, self.mu, self.generator_learning_rate) <= 0:
            raise ConfigError("adamf_mat.margin, epsilon, adversarial_temperature, mu and "
                              "generator_learning_rate must be > 0; switch MAT off with mat: false")
        if "structure" not in self.modalities or tuple(m for m in MODALITIES if m in self.modalities) != tuple(
                self.modalities):
            raise ConfigError(f"adamf_mat.modalities must keep 'structure' and follow the order {MODALITIES}")
        # The generators make the image and text embeddings from the structure one.
        if self.mat and tuple(self.modalities) != MODALITIES:
            raise ConfigError("adamf_mat.mat needs all three modalities; drop a modality with mat: false")

    def run_name(self) -> str:
        """Name of training runs, e.g. `adamf-mat-mu1`, `adamf`, `mean-mat-mu1`, `adamf-no-image` or `rotate`."""
        if tuple(self.modalities) == ("structure",):
            return "rotate"
        fusion = "adamf" if self.adaptive_fusion else "mean"
        name = f"{fusion}-mat-mu{self.mu:g}" if self.mat else fusion
        return name + "".join(f"-no-{m}" for m in MODALITIES if m not in self.modalities)


def sigmoid_loss(positive: torch.Tensor, negative: torch.Tensor, temperature: float) -> torch.Tensor:
    """Self-adversarial negative-sampling loss of RotatE.

    `positive` is (batch, 1), `negative` (batch, negatives). Harder negatives weigh more; the
    weights are not differentiated.
    """
    log_sigmoid = nn.functional.logsigmoid
    weights = torch.softmax(negative * temperature, dim=-1).detach()
    return -(log_sigmoid(positive).mean() + (weights * log_sigmoid(-negative)).sum(dim=-1).mean()) / 2


class ModalityGenerator(nn.Module):
    """Structure embedding and noise → a synthetic embedding of one modality."""

    def __init__(self, config: KgAdamfMatConfig) -> None:
        super().__init__()
        self.noise_dim = config.noise_dim
        self.layers = nn.Sequential(
            nn.Linear(config.noise_dim + 2 * config.dim, config.generator_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.generator_hidden_dim, 2 * config.dim),
        )

    def forward(self, structure: torch.Tensor) -> torch.Tensor:
        """One synthetic embedding per row; a new noise draw on every call."""
        noise = torch.randn(structure.shape[0], self.noise_dim, device=structure.device)
        result: torch.Tensor = self.layers(torch.cat([noise, structure], dim=-1))
        return result


class KgAdamfMat(nn.Module):
    """Scores triples of entity ids; higher means more plausible."""

    image_vectors: torch.Tensor
    text_vectors: torch.Tensor

    def __init__(self, config: KgAdamfMatConfig, n_entities: int, n_relations: int, image_vectors: torch.Tensor,
                 text_vectors: torch.Tensor) -> None:
        super().__init__()
        if image_vectors.shape[0] != n_entities or text_vectors.shape[0] != n_entities:
            raise DataError("image and text vectors need one row per entity")
        self.config = config
        self.n_entities = n_entities
        entity_dim = 2 * config.dim
        self.entities = nn.Embedding(n_entities, entity_dim)
        self.relations = nn.Embedding(n_relations, config.dim)
        # Initial ranges of RotatE; a relation value spans one full turn over its range.
        self.relation_range = (config.margin + config.epsilon) / config.dim
        nn.init.uniform_(self.entities.weight, -(config.margin + config.epsilon) / entity_dim,
                         (config.margin + config.epsilon) / entity_dim)
        nn.init.uniform_(self.relations.weight, -self.relation_range, self.relation_range)
        # Frozen lookup tables; restored from the embedding store, not saved in checkpoints.
        self.register_buffer("image_vectors", image_vectors.float(), persistent=False)
        self.register_buffer("text_vectors", text_vectors.float(), persistent=False)
        if "image" in config.modalities:
            self.image_proj = nn.Linear(image_vectors.shape[1], entity_dim)
        if "text" in config.modalities:
            self.text_proj = nn.Linear(text_vectors.shape[1], entity_dim)
        self.attention = nn.Linear(entity_dim, 1, bias=False)
        if config.mat:
            self.image_generator = ModalityGenerator(config)
            self.text_generator = ModalityGenerator(config)

    def modality_embeddings(self, ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Embedding of each used modality for the given entities, in the order of `MODALITIES`."""
        embeddings = [self.entities(ids)]
        if "image" in self.config.modalities:
            embeddings.append(self.image_proj(self.image_vectors[ids]))
        if "text" in self.config.modalities:
            embeddings.append(self.text_proj(self.text_vectors[ids]))
        return tuple(embeddings)

    def modality_weights(self, *embeddings: torch.Tensor) -> torch.Tensor:
        """Weight of each modality per entity, shape (entities, modalities), in the order given."""
        stacked = torch.stack(embeddings, dim=1)
        if not self.config.adaptive_fusion:
            return torch.full(stacked.shape[:2], 1.0 / stacked.shape[1], device=stacked.device)
        return torch.softmax(self.attention(torch.tanh(stacked)).squeeze(-1), dim=-1)

    def fuse(self, *embeddings: torch.Tensor) -> torch.Tensor:
        """Joint embedding: the weighted sum of the modality embeddings."""
        weights = self.modality_weights(*embeddings).unsqueeze(-1)
        fused: torch.Tensor = (weights * torch.stack(embeddings, dim=1)).sum(dim=1)
        return fused

    def joint(self, ids: torch.Tensor) -> torch.Tensor:
        """Joint embedding of the given entities, shape (entities, 2 * dim)."""
        return self.fuse(*self.modality_embeddings(ids))

    def synthetic_joint(self, ids: torch.Tensor) -> torch.Tensor:
        """Joint embedding with the image and text embeddings replaced by generated ones."""
        structure = self.entities(ids)
        return self.fuse(structure, self.image_generator(structure), self.text_generator(structure))

    def score(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        """Margin minus the RotatE distance; `heads` and `tails` are joint embeddings.

        Shapes broadcast: heads (batch, 1 or n, 2 * dim), tails likewise, relations (batch,).
        """
        phase = self.relations(relations) / (self.relation_range / math.pi)
        re_r, im_r = phase.cos().unsqueeze(1), phase.sin().unsqueeze(1)
        re_h, im_h = heads.chunk(2, dim=-1)
        re_t, im_t = tails.chunk(2, dim=-1)
        re_d = re_h * re_r - im_h * im_r - re_t
        im_d = re_h * im_r + im_h * re_r - im_t
        distance: torch.Tensor = torch.sqrt(re_d.square() + im_d.square() + 1e-12).sum(dim=-1)
        return self.config.margin - distance

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        """Everything but the generators."""
        for name, parameter in self.named_parameters():
            if not name.startswith(("image_generator", "text_generator")):
                yield parameter

    def generator_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters of the two modality generators; none when MAT is off."""
        if self.config.mat:
            yield from self.image_generator.parameters()
            yield from self.text_generator.parameters()

    def _synthetic_scores(self, triples: torch.Tensor) -> list[torch.Tensor]:
        """Scores of each true triple with a synthetic head, a synthetic tail, and both; each (batch, 1)."""
        heads, relations, tails = triples.unbind(dim=1)
        h, t = self.joint(heads).unsqueeze(1), self.joint(tails).unsqueeze(1)
        fake_h, fake_t = self.synthetic_joint(heads).unsqueeze(1), self.synthetic_joint(tails).unsqueeze(1)
        return [self.score(fake_h, relations, t), self.score(h, relations, fake_t),
                self.score(fake_h, relations, fake_t)]

    def encoder_loss(self, triples: torch.Tensor, negatives: torch.Tensor, corrupt_head: bool) -> torch.Tensor:
        """Negative-sampling loss, plus mu times the loss of losing to the synthetic triples.

        `negatives` is (batch, negatives) entity ids that replace the head or the tail.
        """
        heads, relations, tails = triples.unbind(dim=1)
        # One pass over every entity is cheaper than projecting batch × negatives rows.
        everything = self.joint(torch.arange(self.n_entities, device=triples.device))
        h, t = everything[heads].unsqueeze(1), everything[tails].unsqueeze(1)
        corrupted = everything[negatives]
        positive = self.score(h, relations, t)
        negative = self.score(corrupted, relations, t) if corrupt_head else self.score(h, relations, corrupted)
        temperature = self.config.adversarial_temperature
        loss = sigmoid_loss(positive, negative, temperature)
        if self.config.mat:
            for synthetic in self._synthetic_scores(triples):
                loss = loss + self.config.mu * sigmoid_loss(positive, synthetic, temperature)
        return loss

    def generator_loss(self, triples: torch.Tensor) -> torch.Tensor:
        """Loss of the synthetic triples scoring below the true ones; only the generators are stepped on it."""
        heads, relations, tails = triples.unbind(dim=1)
        with torch.no_grad():
            positive = self.score(self.joint(heads).unsqueeze(1), relations, self.joint(tails).unsqueeze(1))
        temperature = self.config.adversarial_temperature
        losses = [sigmoid_loss(synthetic, positive, temperature) for synthetic in self._synthetic_scores(triples)]
        return torch.stack(losses).sum()

    @torch.no_grad()
    def candidate_scores(self, anchors: torch.Tensor, relations: torch.Tensor, anchor_is_head: torch.Tensor,
                         candidates: torch.Tensor) -> torch.Tensor:
        """(rows × candidates) scores of every candidate completing (anchor, relation, ?) or (?, relation, anchor)."""
        pool = self.joint(candidates).unsqueeze(0)
        fixed = self.joint(anchors).unsqueeze(1)
        as_tail = self.score(fixed, relations, pool)
        as_head = self.score(pool, relations, fixed)
        return torch.where(anchor_is_head.unsqueeze(1), as_tail, as_head)
