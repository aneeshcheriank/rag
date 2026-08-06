EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_PATH = "./models"

API_KEY = "DEEPSEEK_API_KEY"

REDIS_URL = "redis://localhost:6379"
REDIS_NAMESPACE = "parent_docs"

CHROMA_STORAGE = "./chroma_db"  # persistence storage for chroma_db
COLLECTION_NAME = "child_docs"  # collection name for chroma_db

LLM_MODEL = "deepseek-v4-flash"

# Model used by DeepEval for metric computation (must support JSON structured output).
# deepseek-chat handles this reliably; deepseek-v4-flash does not.
EVAL_MODEL = "deepseek-v4-pro"

# no of chunks to retrieve from the vectorstore for each query
TOP_K = 2
