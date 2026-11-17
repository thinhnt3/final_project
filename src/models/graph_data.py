"""Turn one image's graph into a PyTorch Geometric `Data` object. No file I/O here.

Layout of a sub-graph: node 0 is the image; then one object node per concept in the image;
then one relation node per triplet, carrying the predicate. Edges are directed:
image → object, subject → relation, relation → object. A relation stays an edge between two
objects in meaning; the relation node only exists so that a GraphSAGE layer, which cannot
read edge labels, can read the predicate.
"""
from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from typing import Any

import torch
from torch_geometric.data import Data

from src.core.exceptions import DataError

IMAGE_NODE = 0


class NodeKind(IntEnum):
    """What a node stands for; decides which embedding table its feature comes from."""

    IMAGE = 0
    OBJECT = 1
    RELATION = 2


def image_graph_to_data(image_graph: Mapping[str, Any], image_row: int) -> Data:
    """Build the sub-graph of one image.

    `feature_row` holds, per node, the row to read in the table of its kind: the image row
    for the image node, the concept id for an object, the predicate id for a relation. The
    name avoids "index" on purpose: PyG shifts every attribute named `*index*` by the node
    offset when it batches graphs, which would corrupt these table rows.
    """
    concept_ids = [int(c) for c in image_graph["concept_ids"]]
    triplets = [(int(s), int(p), int(o)) for s, p, o in image_graph["triplets"]]
    if len(set(concept_ids)) != len(concept_ids):
        raise DataError(f"image {image_graph['vg_image_id']}: repeated concept ids")
    node_of_concept = {concept: 1 + position for position, concept in enumerate(concept_ids)}
    unknown = [t for t in triplets if t[0] not in node_of_concept or t[2] not in node_of_concept]
    if unknown:
        raise DataError(f"image {image_graph['vg_image_id']}: triplets use concepts not in the image: {unknown[:3]}")

    first_relation = 1 + len(concept_ids)
    kinds = [NodeKind.IMAGE] + [NodeKind.OBJECT] * len(concept_ids) + [NodeKind.RELATION] * len(triplets)
    feature_row = [image_row, *concept_ids, *(predicate for _, predicate, _ in triplets)]
    sources = [IMAGE_NODE] * len(concept_ids)
    targets = list(node_of_concept.values())
    for position, (subject, _, target) in enumerate(triplets):
        relation = first_relation + position
        sources += [node_of_concept[subject], relation]
        targets += [relation, node_of_concept[target]]

    return Data(
        edge_index=torch.tensor([sources, targets], dtype=torch.long).reshape(2, -1),
        node_kind=torch.tensor([int(k) for k in kinds], dtype=torch.long),
        feature_row=torch.tensor(feature_row, dtype=torch.long),
        image_id=int(image_graph["vg_image_id"]),
        num_nodes=len(kinds),
    )
