import numpy as np
import pytest

from scrydb import Index


DIM = 16


def _vec(seed: int) -> list:
    rng = np.random.RandomState(seed)
    return rng.uniform(-1, 1, DIM).astype(np.float32).tolist()


@pytest.fixture
def index(tmp_path):
    db_path = tmp_path / "idx.db"
    with Index.open(db_path) as idx:
        docs = [
            {
                "id": f"d{i}",
                "text": f"cats and dogs number {i}" if i % 2 == 0 else f"cars and planes number {i}",
                "emb": _vec(i),
            }
            for i in range(12)
        ]
        idx.index_documents(docs, store_int8_embeddings=True)
        idx.index_queries(
            [{"id": "q0", "text": "cats and dogs", "emb": docs[0]["emb"]}],
            store_int8_embeddings=True,
        )
        yield idx


def test_lexical_search_finds_matching_terms(index):
    results = index.search("cats dogs", mode="lexical", top_k=5)
    assert results
    assert all("cats and dogs" in r.document["text"] for r in results)


def test_lexical_search_empty_query_returns_nothing(index):
    assert index.search("???", mode="lexical") == []


@pytest.mark.parametrize("precision", ["binary", "int8", "float"])
def test_semantic_search_ranks_query_id_first(index, precision):
    run = index.batch_search(mode="semantic", precision=precision)
    results = run["q0"]
    assert results[0].id == "d0"


def test_semantic_rerank_of_itself_raises(index):
    with pytest.raises(ValueError):
        index.batch_search(mode="semantic", precision="binary", rerank="binary")


def test_binary_rerank_with_float_adds_cosine_similarity(index):
    run = index.batch_search(mode="semantic", precision="binary", rerank="float")
    top = run["q0"][0]
    assert top.id == "d0"
    assert "hamming_distance" in top
    assert "cosine_similarity" in top
    assert top.cosine_similarity == pytest.approx(1.0, abs=1e-4)


def test_legacy_rerank_aliases_match_new_names(index):
    legacy = index.batch_search(mode="lexical", rerank="hamming")
    modern = index.batch_search(mode="lexical", rerank="binary")
    assert [r.id for r in legacy["q0"]] == [r.id for r in modern["q0"]]

    legacy = index.batch_search(mode="lexical", rerank="cosine")
    modern = index.batch_search(mode="lexical", rerank="float")
    assert [r.id for r in legacy["q0"]] == [r.id for r in modern["q0"]]


def test_hybrid_search_combines_lexical_and_semantic(index):
    run = index.batch_search(mode="hybrid", rerank=True)
    results = run["q0"]
    assert results
    assert all("rrf_score" in r for r in results)


def test_reindexing_a_document_does_not_duplicate_its_embedding(index):
    before = len(index.document_embeddings)
    index.index_documents([{"id": "d0", "text": "cats and dogs number 0 updated", "emb": _vec(0)}])
    assert len(index.document_embeddings) == before


def test_embedding_properties_round_trip(index):
    assert index.document_embeddings["d0"].dtype == np.float32
    assert index.document_embeddings["d0"].shape == (DIM,)
    assert index.document_embeddings_binary["d0"].dtype == np.uint8
    assert index.document_embeddings_int8["d0"].dtype == np.int8
    assert index.document_embeddings_int8["d0"].shape == (DIM,)

    with pytest.raises(KeyError):
        index.document_embeddings["does-not-exist"]


def test_document_embeddings_int8_empty_when_not_stored(tmp_path):
    with Index.open(tmp_path / "idx.db") as idx:
        idx.index_documents([{"id": "1", "text": "hello", "emb": _vec(0)}])
        assert len(idx.document_embeddings_int8) == 0
        with pytest.raises(KeyError):
            idx.document_embeddings_int8["1"]


def test_semantic_search_without_model_or_embedding_raises(tmp_path):
    with Index.open(tmp_path / "idx.db") as idx:
        idx.index_documents([{"id": "1", "text": "hello", "emb": _vec(0)}])
        with pytest.raises(RuntimeError):
            idx.search("some ad-hoc text", mode="semantic")


def test_write_trec_and_to_dataframe(index, tmp_path):
    run = index.batch_search(mode="lexical")
    out = tmp_path / "run.trec"
    run.write_trec(out, tag="test")
    lines = out.read_text().strip().splitlines()
    assert lines
    for line in lines:
        qid, q0, docid, rank, score, tag = line.split()
        assert q0 == "Q0"
        assert tag == "test"

    pd = pytest.importorskip("pandas")
    df = run.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert {"query_id", "rank", "id"}.issubset(df.columns)
