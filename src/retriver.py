from langchain_community.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.storage import RedisStore
from langchain_classic.storage import create_kv_docstore

import logging
import os

from src.model import get_embeddings
from src.config import CHROMA_STORAGE, REDIS_URL, REDIS_NAMESPACE
from src.data_process import get_splitter

logger = logging.getLogger(__name__)


def get_vectorstore(k: int = 4):
    embedding_model = get_embeddings()
    child_splitter, parent_splitter = get_splitter()

    if not os.path.exists(CHROMA_STORAGE):
        raise FileNotFoundError("No chroma_db found in path {CHROMA_STORAGE}")

    vectorstore = Chroma(
        persist_directory=CHROMA_STORAGE, 
        embedding_function=embedding_model
    )

    logger.info(f"Loded persistent storage from {CHROMA_STORAGE}")

    # initialize the persistent docstore
    try:
        redis_store = RedisStore(redis_url=REDIS_URL, namespace=REDIS_NAMESPACE)
        docstore = create_kv_docstore(redis_store)
    except Exception as e:
        logger.error(f"Failed to connect to Redis at {REDIS_URL}: {e}")
        raise RuntimeError(f"Could not connect to Redis docstore: {e}") from e

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        k=k,
    )
    
    return retriever
