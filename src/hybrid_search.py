import numpy as np
import math
from collections import Counter
from src.logger import get_logger
from src.exceptions import RetrievalError
from config.settings import TOP_K_RESULTS

logger = get_logger("hybrid_search")

class BM25:
    """BM25 implementation from scratch — no external library needed"""
    
    def __init__(self, documents: list, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.tokenized = [doc.lower().split() 
                         for doc in documents]
        self.doc_count = len(documents)
        self.avg_doc_len = sum(len(d) for d in self.tokenized) / max(self.doc_count, 1)
        self.df = self._compute_df()
        
    def _compute_df(self) -> dict:
        """Compute document frequency for each term"""
        df = {}
        for doc in self.tokenized:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        return df
    
    def _idf(self, term: str) -> float:
        """Compute inverse document frequency"""
        df = self.df.get(term, 0)
        return math.log(
            (self.doc_count - df + 0.5) / (df + 0.5) + 1
        )
    
    def get_scores(self, query_terms: list) -> list:
        """Compute BM25 scores for all documents"""
        scores = []
        for doc_tokens in self.tokenized:
            doc_len = len(doc_tokens)
            tf_counts = Counter(doc_tokens)
            score = 0.0
            for term in query_terms:
                tf = tf_counts.get(term, 0)
                idf = self._idf(term)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / 
                    max(self.avg_doc_len, 1)
                )
                score += idf * (numerator / max(denominator, 1))
            scores.append(score)
        return scores


class HybridSearcher:
    def __init__(self, chunks: list, metadatas: list):
        """Initialize hybrid searcher with chunks"""
        self.chunks = chunks
        self.metadatas = metadatas
        self.bm25 = BM25(chunks)
        logger.info(
            f"Hybrid searcher initialized "
            f"with {len(chunks)} chunks"
        )

    def semantic_search(self, model, collection,
                        query: str, top_k: int) -> list:
        """Perform semantic search using ChromaDB"""
        try:
            query_vector = model.encode([query]).tolist()
            results = collection.query(
                query_embeddings=query_vector,
                n_results=top_k
            )
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            logger.info(
                f"Semantic search returned {len(docs)} chunks"
            )
            return list(zip(docs, metas))
        except Exception as e:
            raise RetrievalError(
                f"Semantic search failed: {e}"
            )

    def keyword_search(self, query: str,
                       top_k: int) -> list:
        """Perform BM25 keyword search"""
        try:
            query_terms = query.lower().split()
            scores = self.bm25.get_scores(query_terms)
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append((
                        self.chunks[idx],
                        self.metadatas[idx]
                    ))
            logger.info(
                f"Keyword search returned {len(results)} chunks"
            )
            return results
        except Exception as e:
            raise RetrievalError(
                f"Keyword search failed: {e}"
            )

    def hybrid_search(self, model, collection,
                      query: str,
                      top_k: int = TOP_K_RESULTS) -> list:
        """Combine semantic and keyword search using RRF"""
        try:
            semantic_results = self.semantic_search(
                model, collection, query, top_k
            )
            keyword_results = self.keyword_search(
                query, top_k
            )

            # Reciprocal Rank Fusion
            scores = {}
            k = 60

            for rank, (doc, meta) in enumerate(
                semantic_results
            ):
                key = doc[:100]
                if key not in scores:
                    scores[key] = {
                        "doc": doc, "meta": meta, "score": 0
                    }
                scores[key]["score"] += 1 / (k + rank + 1)

            for rank, (doc, meta) in enumerate(
                keyword_results
            ):
                key = doc[:100]
                if key not in scores:
                    scores[key] = {
                        "doc": doc, "meta": meta, "score": 0
                    }
                scores[key]["score"] += 1 / (k + rank + 1)

            sorted_results = sorted(
                scores.values(),
                key=lambda x: x["score"],
                reverse=True
            )[:top_k]

            chunks = [r["doc"] for r in sorted_results]
            metas = [r["meta"] for r in sorted_results]

            logger.info(
                f"Hybrid search returned {len(chunks)} chunks"
            )
            return chunks, metas

        except Exception as e:
            raise RetrievalError(
                f"Hybrid search failed: {e}"
            )