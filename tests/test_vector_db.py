import pytest
from unittest.mock import patch, MagicMock

from redis.exceptions import ConnectionError as RedisConnectionError

from src.vector_db import parent_document_store


@pytest.fixture
def mock_deps():
    """Patch all heavy dependencies of parent_document_store and return them."""
    with (
        patch("src.vector_db.get_splitter") as mock_splitter,
        patch("src.vector_db.get_embeddings") as mock_emb,
        patch("src.vector_db.Chroma") as mock_chroma,
        patch("src.vector_db.RedisStore") as mock_redis,
        patch("src.vector_db.create_kv_docstore") as mock_docstore,
        patch("src.vector_db.ParentDocumentRetriever") as mock_pdr,
        patch("src.vector_db.shutil.rmtree") as mock_rmtree,
        patch("src.vector_db.os.path.exists") as mock_exists,
    ):
        mock_splitter.return_value = (MagicMock(), MagicMock())
        mock_emb.return_value = MagicMock()

        redis_instance = MagicMock()
        redis_instance.yield_keys.return_value = []
        mock_redis.return_value = redis_instance

        yield {
            "splitter": mock_splitter,
            "embeddings": mock_emb,
            "chroma": mock_chroma,
            "redis": mock_redis,
            "docstore": mock_docstore,
            "pdr": mock_pdr,
            "rmtree": mock_rmtree,
            "exists": mock_exists,
            "redis_instance": redis_instance,
        }


class TestParentDocumentStore:
    def test_returns_retriever_and_adds_documents(self, mock_deps):
        mock_deps["exists"].return_value = False
        docs = [MagicMock(), MagicMock()]
        mock_pdr = mock_deps["pdr"].return_value

        result = parent_document_store(docs, clear_existing=False)

        assert result is mock_pdr
        mock_pdr.add_documents.assert_called_once_with(docs)

    def test_clears_chroma_when_exists_and_clear_requested(self, mock_deps):
        mock_deps["exists"].return_value = True

        parent_document_store([], clear_existing=True)

        mock_deps["rmtree"].assert_called_once()

    def test_does_not_clear_chroma_when_path_missing(self, mock_deps):
        mock_deps["exists"].return_value = False

        parent_document_store([], clear_existing=True)

        mock_deps["rmtree"].assert_not_called()

    def test_does_not_clear_chroma_when_clear_existing_false(self, mock_deps):
        mock_deps["exists"].return_value = True

        parent_document_store([], clear_existing=False)

        mock_deps["rmtree"].assert_not_called()

    def test_clears_redis_keys_when_present(self, mock_deps):
        mock_deps["exists"].return_value = False
        mock_deps["redis_instance"].yield_keys.return_value = ["k1", "k2"]

        parent_document_store([], clear_existing=True)

        mock_deps["redis_instance"].mdelete.assert_called_once_with(["k1", "k2"])

    def test_skips_redis_clear_when_no_keys(self, mock_deps):
        mock_deps["exists"].return_value = False
        mock_deps["redis_instance"].yield_keys.return_value = []

        parent_document_store([], clear_existing=True)

        mock_deps["redis_instance"].mdelete.assert_not_called()

    def test_does_not_clear_redis_when_clear_existing_false(self, mock_deps):
        mock_deps["exists"].return_value = False
        mock_deps["redis_instance"].yield_keys.return_value = ["k1"]

        parent_document_store([], clear_existing=False)

        mock_deps["redis_instance"].mdelete.assert_not_called()

    def test_raises_on_redis_connection_error(self, mock_deps):
        mock_deps["exists"].return_value = False
        mock_deps["redis"].side_effect = RedisConnectionError("down")

        with pytest.raises(RuntimeError, match="Redis"):
            parent_document_store([], clear_existing=False)

    def test_raises_on_unexpected_redis_error(self, mock_deps):
        mock_deps["exists"].return_value = False
        mock_deps["redis"].side_effect = RuntimeError("something else")

        with pytest.raises(RuntimeError, match="Docstore initialization failed"):
            parent_document_store([], clear_existing=False)

    def test_raises_on_redis_clear_failure(self, mock_deps):
        mock_deps["exists"].return_value = False
        mock_deps["redis_instance"].yield_keys.side_effect = Exception("clear failed")

        with pytest.raises(RuntimeError, match="clear Redis docstore"):
            parent_document_store([], clear_existing=True)

    def test_creates_chroma_with_expected_config(self, mock_deps):
        mock_deps["exists"].return_value = False

        parent_document_store([], clear_existing=False)

        mock_deps["chroma"].assert_called_once()
        chroma_kwargs = mock_deps["chroma"].call_args.kwargs
        assert "collection_name" in chroma_kwargs
        assert "persist_directory" in chroma_kwargs
        assert "embedding_function" in chroma_kwargs
