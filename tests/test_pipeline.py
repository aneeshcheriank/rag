import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.pipeline import format_docs, rag


# ---------------------------------------------------------------------------
# format_docs
# ---------------------------------------------------------------------------
class TestFormatDocs:
    def test_joins_multiple_documents_with_double_newline(self):
        docs = [
            Document(page_content="First chunk"),
            Document(page_content="Second chunk"),
            Document(page_content="Third chunk"),
        ]
        result = format_docs(docs)
        assert result == "First chunk\n\nSecond chunk\n\nThird chunk"

    def test_single_document_no_separator(self):
        docs = [Document(page_content="Only chunk")]
        result = format_docs(docs)
        assert result == "Only chunk"

    def test_empty_list_returns_empty_string(self):
        result = format_docs([])
        assert result == ""

    def test_handles_empty_page_content(self):
        docs = [
            Document(page_content="Valid"),
            Document(page_content=""),
            Document(page_content="Also valid"),
        ]
        result = format_docs(docs)
        assert "Valid\n\n" in result
        assert "\n\nAlso valid" in result


# ---------------------------------------------------------------------------
# rag
# ---------------------------------------------------------------------------
class TestRag:
    @pytest.fixture(autouse=True)
    def clear_caches(self):
        """Clear LRU caches so each test gets a fresh retriever/embeddings."""
        from src.retriever import _build_hybrid_child_retriever
        from src.model import get_embeddings

        _build_hybrid_child_retriever.cache_clear()
        get_embeddings.cache_clear()
        yield
        _build_hybrid_child_retriever.cache_clear()
        get_embeddings.cache_clear()

    @patch("src.pipeline.rag_prompt")
    @patch("src.pipeline.get_llm")
    @patch("src.pipeline.get_vectorstore")
    def test_returns_dict_with_response_and_context(
        self, mock_retriever, mock_llm, mock_prompt
    ):
        fake_doc = Document(page_content="Relevant text", metadata={})
        mock_retriever.return_value.invoke.return_value = [fake_doc]
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Generated answer"
        mock_prompt.__or__.return_value.__or__.return_value = mock_chain

        result = rag("What is Apple's revenue?")

        assert isinstance(result, dict)
        assert "response" in result
        assert "context" in result
        assert result["response"] == "Generated answer"
        assert result["context"] == [fake_doc]

    @patch("src.pipeline.rag_prompt")
    @patch("src.pipeline.get_llm")
    @patch("src.pipeline.get_vectorstore")
    def test_passes_k_to_retriever(self, mock_retriever, mock_llm, mock_prompt):
        mock_retriever.return_value.invoke.return_value = []
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = ""
        mock_prompt.__or__.return_value.__or__.return_value = mock_chain

        rag("query", k=5)

        mock_retriever.assert_called_once_with(k=5)

    @patch("src.pipeline.rag_prompt")
    @patch("src.pipeline.get_llm")
    @patch("src.pipeline.get_vectorstore")
    def test_default_k_is_8(self, mock_retriever, mock_llm, mock_prompt):
        mock_retriever.return_value.invoke.return_value = []
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = ""
        mock_prompt.__or__.return_value.__or__.return_value = mock_chain

        rag("query")

        mock_retriever.assert_called_once_with(k=8)

    @patch("src.pipeline.rag_prompt")
    @patch("src.pipeline.get_llm")
    @patch("src.pipeline.get_vectorstore")
    def test_invokes_chain_with_context(self, mock_retriever, mock_llm, mock_prompt):
        doc1 = Document(page_content="Chunk A")
        doc2 = Document(page_content="Chunk B")
        mock_retriever.return_value.invoke.return_value = [doc1, doc2]

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Answer"
        mock_prompt.__or__.return_value.__or__.return_value = mock_chain

        rag("question")

        call_args = mock_chain.invoke.call_args[0][0]
        assert call_args["context"] == "Chunk A\n\nChunk B"
        assert call_args["question"] == "question"

    @patch("src.pipeline.rag_prompt")
    @patch("src.pipeline.get_llm")
    @patch("src.pipeline.get_vectorstore")
    def test_passes_chat_history_to_chain(self, mock_retriever, mock_llm, mock_prompt):
        mock_retriever.return_value.invoke.return_value = []
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = ""
        mock_prompt.__or__.return_value.__or__.return_value = mock_chain

        history = [{"role": "user", "content": "previous"}]
        rag("query", chat_history=history)

        call_args = mock_chain.invoke.call_args[0][0]
        assert call_args["chat_history"] == history

    @patch("src.pipeline.rag_prompt")
    @patch("src.pipeline.get_llm")
    @patch("src.pipeline.get_vectorstore")
    def test_handles_empty_retrieval_context(
        self, mock_retriever, mock_llm, mock_prompt
    ):
        mock_retriever.return_value.invoke.return_value = []
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "I don't know"
        mock_prompt.__or__.return_value.__or__.return_value = mock_chain

        result = rag("obscure question")

        assert result["context"] == []
        assert result["response"] == "I don't know"
