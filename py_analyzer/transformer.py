from functools import lru_cache
from sentence_transformers import SentenceTransformer


class Transformer:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    @lru_cache(maxsize=32)
    def calc_embeddings(self, sentences):
        return self.model.encode(sentences)
