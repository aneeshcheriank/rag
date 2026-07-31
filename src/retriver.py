from langchain_chroma import Chroma

import logging
import os

from src.model import get_embeddings
from src.config import CHROMA_STORAGE

logger = logging.getLogger(__name__)


def get_vectorstore(k: int = 4):
    embedding_model = get_embeddings()

    if not os.path.exists(CHROMA_STORAGE):
        raise FileNotFoundError("No chroma_db found in path {CHROMA_STORAGE}")

    vectorstore = Chroma(
        persist_directory=CHROMA_STORAGE, embedding_function=embedding_model
    )

    logger.info(f"Loded persistent storage from {CHROMA_STORAGE}")

    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Options: "similarity", "mmr", "similarity_score_threshold"
        search_kwargs={"k": k},
    )

    return retriever
