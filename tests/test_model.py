import pytest
from unittest.mock import patch, MagicMock
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr


# ---------------------------------------------------------------------------
# get_embeddings
# ---------------------------------------------------------------------------
class TestGetEmbeddings:
    @pytest.fixture(autouse=True)
    def clear_lru_cache(self):
        """Clear the @lru_cache on get_embeddings before each test."""
        from src.model import get_embeddings

        get_embeddings.cache_clear()
        yield
        get_embeddings.cache_clear()

    @patch("src.model.torch.cuda.is_available", return_value=False)
    @patch("src.model.os.path.exists", return_value=True)
    def test_returns_huggingface_embeddings_cpu(self, mock_exists, mock_cuda):
        from src.model import get_embeddings

        emb = get_embeddings()
        assert isinstance(emb, HuggingFaceEmbeddings)

    @patch("src.model.torch.cuda.is_available", return_value=False)
    @patch("src.model.os.path.exists", return_value=True)
    def test_device_is_cpu_when_cuda_unavailable(self, mock_exists, mock_cuda):
        from src.model import get_embeddings

        emb = get_embeddings()
        assert emb.model_kwargs["device"] == "cpu"

    @patch("src.model.torch.cuda.is_available", return_value=True)
    @patch("src.model.os.path.exists", return_value=True)
    @patch("src.model.HuggingFaceEmbeddings")
    def test_device_is_cuda_when_available(self, mock_hf, mock_exists, mock_cuda):
        from src.model import get_embeddings

        get_embeddings()
        kwargs = mock_hf.call_args.kwargs
        assert kwargs["model_kwargs"]["device"] == "cuda"

    @patch("src.model.os.path.exists", return_value=True)
    def test_encode_kwargs_normalize_embeddings(self, mock_exists):
        from src.model import get_embeddings

        emb = get_embeddings()
        assert emb.encode_kwargs["normalize_embeddings"] is True

    @patch("src.model.os.path.exists", return_value=True)
    def test_local_files_only_when_model_exists(self, mock_exists):
        from src.model import get_embeddings

        emb = get_embeddings()
        assert emb.model_kwargs["local_files_only"] is True

    @patch("src.model.os.path.exists", return_value=True)
    def test_uses_config_model_path(self, mock_exists):
        from src.model import get_embeddings
        from src import config

        emb = get_embeddings()
        assert emb.model_name == config.EMBEDDING_MODEL_PATH

    @patch("src.model.os.path.exists", return_value=True)
    def test_lru_cache_reuses_instance(self, mock_exists):
        from src.model import get_embeddings

        emb1 = get_embeddings()
        emb2 = get_embeddings()
        assert emb1 is emb2

    @patch("src.model.os.path.exists", return_value=False)
    @patch("src.model.SentenceTransformer")
    def test_downloads_model_when_not_found(self, mock_st, mock_exists):
        from src.model import get_embeddings
        from src import config

        mock_model = MagicMock()
        mock_st.return_value = mock_model

        emb = get_embeddings()

        mock_st.assert_called_once_with(config.EMBEDDING_MODEL)
        mock_model.save.assert_called_once_with(config.EMBEDDING_MODEL_PATH)
        assert isinstance(emb, HuggingFaceEmbeddings)


# ---------------------------------------------------------------------------
# get_llm
# ---------------------------------------------------------------------------
class TestGetLlm:
    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key-123"}, clear=True)
    def test_returns_chat_deepseek_instance(self):
        from src.model import get_llm

        llm = get_llm()
        assert isinstance(llm, ChatDeepSeek)

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key-123"}, clear=True)
    def test_uses_config_model_name(self):
        from src.model import get_llm
        from src import config

        llm = get_llm()
        assert llm.model_name == config.LLM_MODEL

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key-123"}, clear=True)
    def test_temperature_is_zero(self):
        from src.model import get_llm

        llm = get_llm()
        assert llm.temperature == 0

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key-123"}, clear=True)
    def test_api_key_is_secret_str(self):
        from src.model import get_llm

        llm = get_llm()
        secret = llm.api_key
        assert isinstance(secret, SecretStr)
        assert secret.get_secret_value() == "test-key-123"

    @patch.dict(os.environ, {}, clear=True)
    def test_raises_when_api_key_missing(self):
        from src.model import get_llm

        with pytest.raises(ValueError, match="api key is missing"):
            get_llm()
