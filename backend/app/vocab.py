"""The 600-word vocabulary, embedded once, answering nearest-neighbour queries."""

import numpy as np

from app.embed import embed_texts
from data.vocabulary import VOCABULARY


class VocabIndex:
    """Embeds all 600 words once at startup (~2s) and holds the matrix.

    Built once in the app lifespan and reused for every request -- re-embedding
    600 fixed words per request would be 600x the work for no benefit, since the
    vocabulary never changes.
    """

    def __init__(self) -> None:
        self.words: list[str] = []
        self.categories: list[str] = []
        for category, words in VOCABULARY.items():
            for word in words:
                self.words.append(word)
                self.categories.append(category)
        self.matrix = embed_texts(self.words)  # (600, 384) float32, unit rows

    def neighbours(self, text: str, k: int = 5) -> list[dict]:
        """Brute-force cosine over 600 rows. No index, no approximation.

        600 x 384 float32 is 920 KB and one matmul is microseconds, so an ANN index
        here would add a dependency, an approximation, and a recall number to
        explain, in exchange for nothing measurable. The sibling project uses
        Qdrant because it has 2,014 vectors and needs payload filtering; this has
        neither.
        """
        q = embed_texts([text])[0]
        scores = self.matrix @ q                      # rows are unit, q is unit -> cosine
        order = np.argsort(-scores)
        out = []
        for i in order:
            if self.words[i].lower() == text.strip().lower():
                continue                              # a word is its own best neighbour
            out.append({"word": self.words[i], "category": self.categories[i],
                        "score": round(float(scores[i]), 4)})
            if len(out) == k:
                break
        return out
