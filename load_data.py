from src.data_process import load_pdf
from src.vector_db import parent_document_store

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
)

pdf_path = "data/10K.pdf"

if __name__ == "__main__":
    docs = load_pdf(pdf_path)
    retriever = parent_document_store(docs, clear_existing=True)
    logger.info("Vector database created successfully.")
