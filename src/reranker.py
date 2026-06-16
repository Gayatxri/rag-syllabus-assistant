from groq import Groq
from src.logger import get_logger
from src.exceptions import RetrievalError
from config.settings import GROQ_API_KEY, LLM_MODEL

logger = get_logger("reranker")

def score_chunk(client: Groq, question: str,
                chunk: str) -> float:
    """Score a single chunk against the question"""
    try:
        prompt = f"""Rate how relevant this text chunk is 
for answering the question below.

Question: {question}

Text chunk: {chunk[:300]}

Rate relevance from 0.0 to 1.0 where:
0.0 = completely irrelevant
0.5 = somewhat relevant  
1.0 = perfectly answers the question

Reply with ONLY a number between 0.0 and 1.0.
Nothing else."""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )

        score_text = response.choices[0].message.content.strip()

        # Extract float from response
        score = float(score_text.split()[0])
        score = max(0.0, min(1.0, score))  # clamp to [0,1]
        return score

    except Exception as e:
        logger.warning(f"Scoring failed, defaulting to 0.5: {e}")
        return 0.5


def rerank_chunks(question: str, chunks: list,
                  metadatas: list) -> tuple:
    """Rerank chunks by relevance to question"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        logger.info(
            f"Reranking {len(chunks)} chunks "
            f"for question: '{question[:50]}'"
        )

        # Score each chunk
        scored = []
        for i, (chunk, meta) in enumerate(
            zip(chunks, metadatas)
        ):
            score = score_chunk(client, question, chunk)
            scored.append((score, chunk, meta))
            logger.info(
                f"Chunk {i+1} score: {score:.2f} "
                f"(Page {meta.get('page', '?')})"
            )

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        reranked_chunks = [item[1] for item in scored]
        reranked_metas = [item[2] for item in scored]
        scores = [item[0] for item in scored]

        logger.info(
            f"Reranking complete. "
            f"Top score: {scores[0]:.2f}"
        )
        return reranked_chunks, reranked_metas, scores

    except Exception as e:
        raise RetrievalError(f"Reranking failed: {e}")