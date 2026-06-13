from functools import lru_cache
from sentence_transformers import CrossEncoder

@lru_cache(maxsize=1)
def get_reranker():
    # Loads exactly once per process when first called
    return CrossEncoder("BAAI/bge-reranker-base")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"