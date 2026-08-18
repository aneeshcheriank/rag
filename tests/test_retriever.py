import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.retriever import (
    HybridParentRetriever,
    _build_hybrid_child_retriever,
    get_vectorstore,
)


# ---------------------------------------------------------------------------
# HybridParentRetriever
# ---------------------------------------------------------------------------
def _make_retriever(child_docs, docstore_map, k=4):
    """Helper: build a HybridParentRetriever with a mocked child retriever and
    a mocked docstore that resolves parent ids via ``docstore_map``."""
    child_retriever = MagicMock()
    child_retriever.invoke.return_value = child_docs

    docstore = MagicMock()

    def fake_mget(ids):
        return [docstore_map.get(pid) for pid in ids]

    docstore.mget.side_effect = fake_mget

    return HybridParentRetriever(child_retriever=child_retriever, docstore=docstore, k=k)


class TestHybridParentRetriever:
    def test_maps_child_to_parent(self):
        parent = Document(page_content="Parent text", metadata={"id": "p1"})
        child = Document(page_content="Child text", metadata={"doc_id": "p1"})

        retriever = _make_retriever([child], {"p1": parent})

        results = retriever.invoke("query")
        assert results == [parent]

    def test_deduplicates_children_mapping_to_same_parent(self):
        parent = Document(page_content="Parent text", metadata={"id": "p1"})
        child1 = Document(page_content="Child 1", metadata={"doc_id": "p1"})
        child2 = Document(page_content="Child 2", metadata={"doc_id": "p1"})

        retriever = _make_retriever([child1, child2], {"p1": parent})

        results = retriever.invoke("query")
        assert results == [parent]
        assert len(results) == 1

    def test_skips_child_without_doc_id(self):
        parent = Document(page_content="Parent", metadata={"id": "p1"})
        child_no_id = Document(page_content="No id", metadata={})
        child_with_id = Document(page_content="Has id", metadata={"doc_id": "p1"})

        retriever = _make_retriever([child_no_id, child_with_id], {"p1": parent})

        results = retriever.invoke("query")
        assert results == [parent]

    def test_skips_child_whose_parent_is_missing(self):
        child_orphan = Document(page_content="Orphan", metadata={"doc_id": "missing"})
        child_ok = Document(page_content="Ok", metadata={"doc_id": "p1"})
        parent = Document(page_content="Parent", metadata={"id": "p1"})

        retriever = _make_retriever([child_orphan, child_ok], {"p1": parent})

        results = retriever.invoke("query")
        assert results == [parent]

    def test_respects_k_limit(self):
        parents = {
            f"p{i}": Document(page_content=f"Parent {i}", metadata={"id": f"p{i}"})
            for i in range(5)
        }
        children = [
            Document(page_content=f"Child {i}", metadata={"doc_id": f"p{i}"})
            for i in range(5)
        ]

        retriever = _make_retriever(children, parents, k=3)

        results = retriever.invoke("query")
        assert len(results) == 3

    def test_returns_empty_for_no_children(self):
        retriever = _make_retriever([], {}, k=4)
        results = retriever.invoke("query")
        assert results == []

    def test_maps_multiple_distinct_parents(self):
        parents = {
            "p1": Document(page_content="Parent 1"),
            "p2": Document(page_content="Parent 2"),
        }
        children = [
            Document(page_content="Child 1", metadata={"doc_id": "p1"}),
            Document(page_content="Child 2", metadata={"doc_id": "p2"}),
        ]

        retriever = _make_retriever(children, parents, k=4)

        results = retriever.invoke("query")
        assert [d.page_content for d in results] == ["Parent 1", "Parent 2"]

    def test_k_defaults_to_4(self):
        retriever = HybridParentRetriever(
            child_retriever=MagicMock(), docstore=MagicMock()
        )
        assert retriever.k == 4


# ---------------------------------------------------------------------------
# _build_hybrid_child_retriever
# ---------------------------------------------------------------------------
class TestBuildHybridChildRetriever:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        _build_hybrid_child_retriever.cache_clear()
        yield
        _build_hybrid_child_retriever.cache_clear()

    @patch("src.retriever.os.path.exists", return_value=False)
    @patch("src.retriever.get_embeddings")
    def test_raises_when_chroma_missing(self, mock_emb, mock_exists):
        with pytest.raises(FileNotFoundError, match="chroma_db"):
            _build_hybrid_child_retriever(bm25_k=4)

    @patch("src.retriever.os.path.exists", return_value=True)
    @patch("src.retriever.get_embeddings")
    @patch("src.retriever.get_splitter")
    @patch("src.retriever.Chroma")
    @patch("src.retriever.EnsembleRetriever")
    @patch("src.retriever.RedisStore", side_effect=ConnectionError("redis down"))
    def test_wraps_redis_error_as_runtime_error(
        self,
        mock_redis,
        mock_ensemble,
        mock_chroma,
        mock_splitter,
        mock_emb,
        mock_exists,
    ):
        mock_splitter.return_value = (MagicMock(), MagicMock())
        mock_chroma.return_value.get.return_value = {
            "documents": ["some child chunk text"],
            "metadatas": [{"doc_id": "p1"}],
        }

        with pytest.raises(RuntimeError, match="Redis"):
            _build_hybrid_child_retriever(bm25_k=4)


# ---------------------------------------------------------------------------
# get_vectorstore
# ---------------------------------------------------------------------------
class TestGetVectorstore:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        _build_hybrid_child_retriever.cache_clear()
        yield
        _build_hybrid_child_retriever.cache_clear()

    @patch("src.retriever._build_hybrid_child_retriever")
    def test_returns_hybrid_parent_retriever(self, mock_build):
        mock_build.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        retriever = get_vectorstore(k=6, bm25_k=3)

        assert isinstance(retriever, HybridParentRetriever)
        assert retriever.k == 6

    @patch("src.retriever._build_hybrid_child_retriever")
    def test_passes_bm25_k_to_builder(self, mock_build):
        mock_build.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        get_vectorstore(k=6, bm25_k=3)

        mock_build.assert_called_once_with(3)
