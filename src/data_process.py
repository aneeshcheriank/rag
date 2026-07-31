from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import logging

from src.vector_db import vector_store

logger = logging.getLogger(__name__)


def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    doc = loader.load()
    logger.info(f"Loaded the doc from path: {pdf_path}")
    return doc


def split_doc(document):
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    texts = splitter.split_documents(document)
    logger.info(f"chunked docs, lenght: {len(texts)}")
    return texts


def process_pdf(pdf_path):
    doc = load_pdf(pdf_path)
    texts = split_doc(doc)

    vector_db = vector_store(texts, clear_existing=True)

    return vector_db
