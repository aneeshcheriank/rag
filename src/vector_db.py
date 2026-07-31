from langchain_chroma import Chroma

from src.config import CHROMA_STORAGE
from src.model import get_embeddings

import os
import shutil

import logging

logger = logging.getLogger(__name__)

embeddings = get_embeddings()


def vector_store(texts, clear_existing=True):

    if clear_existing:
        if os.path.exists(CHROMA_STORAGE):
            shutil.rmtree(CHROMA_STORAGE)
            logger.info(f"Cleared existing database at {CHROMA_STORAGE}")

    len_text = len(texts)
    logger.info(f"started embeddings")
    vectorstore = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory=CHROMA_STORAGE
    )
    logger.info(f"{len_text} has been encoded and stored in vector db")
    return vectorstore
