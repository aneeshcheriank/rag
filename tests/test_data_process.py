import pytest
from unittest.mock import patch, MagicMock

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.data_process import load_pdf, get_splitter


# ---------------------------------------------------------------------------
# get_splitter
# ---------------------------------------------------------------------------
class TestGetSplitter:
    def test_returns_two_splitters(self):
        child, parent = get_splitter()
        assert isinstance(child, RecursiveCharacterTextSplitter)
        assert isinstance(parent, RecursiveCharacterTextSplitter)

    def test_child_splitter_small_chunks(self):
        child, _ = get_splitter()
        assert child._chunk_size == 250
        assert child._chunk_overlap == 50

    def test_parent_splitter_large_chunks(self):
        _, parent = get_splitter()
        assert parent._chunk_size == 2000
        assert parent._chunk_overlap == 200

    def test_parent_splitter_separators(self):
        _, parent = get_splitter()
        assert parent._separators == ["\n\n", "\n", " ", ""]

    def test_child_chunk_overlap_less_than_size(self):
        child, _ = get_splitter()
        assert child._chunk_overlap < child._chunk_size

    def test_splitters_are_different_objects(self):
        child, parent = get_splitter()
        assert child is not parent


# ---------------------------------------------------------------------------
# load_pdf
# ---------------------------------------------------------------------------
class TestLoadPdf:
    @patch("src.data_process.PyPDFLoader")
    def test_loads_and_returns_documents(self, mock_loader_class):
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            Document(page_content="Page 1 text", metadata={"page": 0}),
            Document(page_content="Page 2 text", metadata={"page": 1}),
        ]
        mock_loader_class.return_value = mock_loader

        docs = load_pdf("/fake/path/file.pdf")

        mock_loader_class.assert_called_once_with("/fake/path/file.pdf")
        mock_loader.load.assert_called_once()
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].page_content == "Page 1 text"

    @patch("src.data_process.PyPDFLoader")
    def test_handles_empty_pdf(self, mock_loader_class):
        mock_loader = MagicMock()
        mock_loader.load.return_value = []
        mock_loader_class.return_value = mock_loader

        docs = load_pdf("/fake/path/empty.pdf")
        assert docs == []

    @patch("src.data_process.PyPDFLoader")
    def test_propagates_file_not_found_error(self, mock_loader_class):
        mock_loader_class.side_effect = FileNotFoundError("No such file")

        with pytest.raises(FileNotFoundError):
            load_pdf("/nonexistent/file.pdf")

    @patch("src.data_process.PyPDFLoader")
    def test_preserves_document_metadata(self, mock_loader_class):
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            Document(page_content="text", metadata={"page": 0, "source": "file.pdf"})
        ]
        mock_loader_class.return_value = mock_loader

        docs = load_pdf("file.pdf")
        assert docs[0].metadata["page"] == 0
        assert docs[0].metadata["source"] == "file.pdf"
