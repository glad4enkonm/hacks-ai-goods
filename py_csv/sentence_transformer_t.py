from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Transformer:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    @lru_cache(maxsize=32)
    def calc_embeddings(self, sentences):
        return self.model.encode(sentences)


def check_sim(embeddings):
    similarity = cosine_similarity(embeddings)
    print(similarity)
