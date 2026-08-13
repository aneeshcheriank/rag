from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.storage import RedisStore
from langchain_classic.storage import create_kv_docstore

import logging
import os
from functools import lru_cache
from typing import Any

from src.model import get_embeddings
from src.config import CHROMA_STORAGE, REDIS_URL, REDIS_NAMESPACE, COLLECTION_NAME
from src.data_process import get_splitter

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _build_hybrid_child_retriever(
    bm25_k: int,
) -> tuple[EnsembleRetriever, object, object, object]:
    """Build & cache the expensive parts: Chroma, BM25 index, Ensemble, docstore.

    Cached separately so that changing ``k`` (a runtime search parameter) does
    NOT trigger a full rebuild.
    """
    embedding_model = get_embeddings()
    child_splitter, parent_splitter = get_splitter()

    if not os.path.exists(CHROMA_STORAGE):
        raise FileNotFoundError(f"No chroma_db found in path {CHROMA_STORAGE}")

    vectorstore = Chroma(
        persist_directory=CHROMA_STORAGE,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )
    logger.info("Loaded Chroma store from %s", CHROMA_STORAGE)

    # Build BM25 sparse retriever from the same child chunks
    logger.info("Building BM25 retriever from child chunks …")
    chroma_data = vectorstore.get(include=["documents", "metadatas"])
    child_chunks = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(chroma_data["documents"], chroma_data["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(child_chunks)
    bm25_retriever.k = bm25_k
    logger.info("BM25 retriever ready")

    # Hybrid: dense + sparse with equal weight
    hybrid_child = EnsembleRetriever(
        retrievers=[bm25_retriever, vectorstore.as_retriever()],
        weights=[0.5, 0.5],
    )

    # Persistent docstore (Redis)
    try:
        redis_store = RedisStore(redis_url=REDIS_URL, namespace=REDIS_NAMESPACE)
        docstore = create_kv_docstore(redis_store)
    except Exception as e:
        logger.error("Failed to connect to Redis at %s: %s", REDIS_URL, e)
        raise RuntimeError(f"Could not connect to Redis docstore: {e}") from e

    return hybrid_child, child_splitter, parent_splitter, docstore


class HybridParentRetriever(BaseRetriever):
    """Custom retriever: hybrid (dense + BM25) child search → parent lookup.

    ``ParentDocumentRetriever`` requires a ``VectorStore``, so we can't pass an
    ``EnsembleRetriever`` directly.  This class does the same parent-mapping
    logic manually on top of an arbitrary child retriever.
    """

    child_retriever: Any
    docstore: Any
    k: int = 4

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> list[Document]:
        # 1. Get child docs from the hybrid retriever
        child_docs: list[Document] = self.child_retriever.invoke(query)

        # 2. Map each child to its parent via the docstore, deduplicating
        seen: set[str] = set()
        parents: list[Document] = []

        for child in child_docs:
            parent_id = child.metadata.get("doc_id")
            if not parent_id or parent_id in seen:
                continue

            parent_doc = self.docstore.mget([parent_id])
            if parent_doc and parent_doc[0] is not None:
                parents.append(parent_doc[0])
                seen.add(parent_id)

            if len(parents) >= self.k:
                break

        return parents[: self.k]


def get_vectorstore(k: int = 8, bm25_k: int = 4):
    """Return a :class:`HybridParentRetriever` that does hybrid (dense + BM25)
    child search with parent-document expansion.

    The expensive build work is cached; only the retriever wrapper is
    constructed per call so that *k* can vary without a rebuild."""
    hybrid_child, _child_splitter, _parent_splitter, docstore = (
        _build_hybrid_child_retriever(bm25_k)
    )

    return HybridParentRetriever(
        child_retriever=hybrid_child,
        docstore=docstore,
        k=k,
    )
