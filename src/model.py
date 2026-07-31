import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from sentence_transformers import SentenceTransformer

import os
import logging
from functools import lru_cache
from dotenv import load_dotenv, find_dotenv
from pydantic import SecretStr

from src import config

logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())


@lru_cache(maxsize=1)
def get_embeddings():

    if not os.path.exists(config.EMBEDDING_MODEL_PATH):
        logger.info(
            f"downloading {config.EMBEDDING_MODEL} to {config.EMBEDDING_MODEL_PATH}"
        )
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        model.save(config.EMBEDDING_MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"device": device, "local_files_only": True}
    encode_kwargs = {"normalize_embeddings": True}

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_PATH,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


def get_llm():

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DeepSeek api key is missing from environment")

    return ChatDeepSeek(
        model=config.LLM_MODEL, temperature=0, api_key=SecretStr(api_key)
    )
