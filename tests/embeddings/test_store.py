import numpy as np
import pytest

from src.core.exceptions import DataError
from src.data.json_io import write_json
from src.data.sources import sha256_of
from src.embeddings.store import (
    META_FILE,
    EmbeddingTable,
    encode_in_batches,
    l2_normalize,
    load_table,
    save_table,
    table_paths,
    verify_inputs_unchanged,
)


def _unit(rows):
    return l2_normalize(np.asarray(rows, dtype=np.float32))


def test_l2_normalize_makes_unit_rows_and_rejects_zero():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, -2.0]]))
    assert out.dtype == np.float32
    assert np.allclose(out, [[0.6, 0.8], [0.0, -1.0]])
    with pytest.raises(DataError, match="zero vector"):
        l2_normalize(np.array([[0.0, 0.0]]))


def test_encode_in_batches_keeps_order_and_batches():
    seen = []

    def encode(batch):
        seen.append(list(batch))
        return np.array([[float(x), 1.0] for x in batch])

    out = encode_in_batches([1, 2, 3, 4, 5], encode, batch_size=2)
    assert seen == [[1, 2], [3, 4], [5]]
    assert out.shape == (5, 2) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert np.allclose(out[2], np.array([3.0, 1.0]) / np.sqrt(10.0))


def test_encode_in_batches_rejects_wrong_shape_and_empty_input():
    with pytest.raises(DataError, match="returned shape"):
        encode_in_batches([1, 2], lambda batch: np.ones((1, 3)), batch_size=2)
    with pytest.raises(DataError, match="nothing to encode"):
        encode_in_batches([], lambda batch: np.ones((0, 3)), batch_size=2)


def test_table_select_returns_rows_in_requested_order():
    table = EmbeddingTable(ids=(10, 20, 30), vectors=_unit([[1, 0], [0, 1], [1, 1]]))
    assert np.allclose(table.select([30, 10]), _unit([[1, 1], [1, 0]]))
    assert table.row_of() == {10: 0, 20: 1, 30: 2}
    with pytest.raises(DataError, match="no embedding"):
        table.select([10, 99])


@pytest.mark.parametrize(
    ("ids", "vectors", "message"),
    [
        ((1, 2), _unit([[1, 0]]), "ids but vectors"),
        ((1, 1), _unit([[1, 0], [0, 1]]), "not unique"),
        ((1,), np.array([[1.0, 0.0]], dtype=np.float64), "float32"),
        ((1,), np.array([[np.nan, 0.0]], dtype=np.float32), "NaN"),
        ((1,), np.array([[2.0, 0.0]], dtype=np.float32), "unit length"),
    ],
)
def test_table_rejects_invalid_content(ids, vectors, message):
    with pytest.raises(DataError, match=message):
        EmbeddingTable(ids=ids, vectors=vectors)


def test_save_and_load_round_trip(tmp_path):
    table = EmbeddingTable(ids=("coco-1", "coco-2"), vectors=_unit([[1, 2], [3, 4]]))
    save_table(tmp_path / "nested", "queries", table)
    loaded = load_table(tmp_path / "nested", "queries")
    assert loaded.ids == table.ids
    assert np.array_equal(loaded.vectors, table.vectors)
    with pytest.raises(DataError, match="not found"):
        load_table(tmp_path / "nested", "images")


def test_encode_in_batches_rejects_non_positive_batch_size():
    with pytest.raises(DataError, match="batch_size"):
        encode_in_batches([1], lambda batch: np.ones((1, 2)), batch_size=0)


def test_stale_embeddings_are_detected(tmp_path):
    source = tmp_path / "queries.json"
    source.write_text("[1]")
    directory = tmp_path / "emb"
    with pytest.raises(DataError, match="compute the embeddings first"):
        verify_inputs_unchanged(directory)
    table = EmbeddingTable(ids=(1,), vectors=_unit([[1, 0]]))
    save_table(directory, "images", table)
    table_hashes = {p.name: sha256_of(p) for p in table_paths(directory, "images")}
    write_json(directory / META_FILE, {
        "inputs_sha256": {str(source): sha256_of(source)},
        "tables": {"images": {"rows": 1, "sha256": table_hashes}},
    })
    verify_inputs_unchanged(directory)  # unchanged input: no error
    save_table(directory, "images", EmbeddingTable(ids=(1,), vectors=_unit([[0, 1]])))
    with pytest.raises(DataError, match="differ from meta.json"):
        verify_inputs_unchanged(directory)
    save_table(directory, "images", table)
    source.write_text("[2]")
    with pytest.raises(DataError, match="stale"):
        verify_inputs_unchanged(directory)
    source.unlink()
    with pytest.raises(DataError, match="stale"):
        verify_inputs_unchanged(directory)
