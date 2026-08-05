from langchain_chroma import Chroma
from langchain_community.storage import RedisStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import create_kv_docstore

from redis.exceptions import ConnectionError as RedisConnectionError, RedisError

from src.config import CHROMA_STORAGE, REDIS_URL, REDIS_NAMESPACE
from src.model import get_embeddings
from src.data_process import get_splitter

import os
import shutil

import logging

logger = logging.getLogger(__name__)


def vector_store(texts, clear_existing=True):

    if clear_existing:
        if os.path.exists(CHROMA_STORAGE):
            shutil.rmtree(CHROMA_STORAGE)
            logger.info(f"Cleared existing database at {CHROMA_STORAGE}")

    len_text = len(texts)
    logger.info(f"started embeddings")

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory=CHROMA_STORAGE
    )
    logger.info(f"{len_text} has been encoded and stored in vector db")
    return vectorstore

def parent_document_store(docs, clear_existing=True):

    if clear_existing:
        if os.path.exists(CHROMA_STORAGE):
            shutil.rmtree(CHROMA_STORAGE)
            logger.info(f"Cleared existing database at {CHROMA_STORAGE}")

    logger.info(f"started embeddings")

    child_splitter, parent_splitter = get_splitter()
    
    embeddings = get_embeddings()
    # vectorstore
    vectorstore = Chroma(
        collection_name="child_docs",
        persist_directory=CHROMA_STORAGE, 
        embedding_function=embeddings
    )

    # Intialize the Docstore (stroe paretnt full texts)
    try:
        redis_store = RedisStore(redis_url=REDIS_URL, namespace=REDIS_NAMESPACE)
        docstore = create_kv_docstore(redis_store)
    except (RedisConnectionError, RedisError) as e:
        logger.error(f"Failed to connect to Redis at {REDIS_URL}: {e}")
        raise RuntimeError(f"Could not connect to Redis docstore: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error initializing Redis docstore: {e}")
        raise RuntimeError(f"Docstore initialization failed: {e}") from e

    if clear_existing:
        try:
            keys = list(redis_store.yield_keys())
            if keys:
                redis_store.mdelete(keys)
            logger.info(f"Cleared existing Redis docstore at {REDIS_URL} with namespace {REDIS_NAMESPACE}")
        except Exception as e:
            logger.error(f"Failed to clear Redis docstore: {e}")
            raise RuntimeError(f"Could not clear Redis docstore: {e}") from e
    
    # Build the ParentDocumentRetriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    retriever.add_documents(docs)
    logger.info("Documents have been added to the Chroma and Redis store")

    return retriever
