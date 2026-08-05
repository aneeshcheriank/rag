from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import logging

logger = logging.getLogger(__name__)


def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    doc = loader.load()
    logger.info(f"Loaded the doc from path: {pdf_path}")
    return doc
