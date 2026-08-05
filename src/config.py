EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_PATH = "./models"

REDIS_URL = "redis://localhost:6379"
REDIS_NAMESPACE = "parent_docs"

CHROMA_STORAGE = "./chroma_db"  # persistence storage for chroma_db
COLLECTION_NAME = "child_docs"  # collection name for chroma_db

LLM_MODEL = "deepseek-v4-flash"

# no of chunks to retrieve from the vectorstore for each query
TOP_K = 2
