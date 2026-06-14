from groq import Groq
from src.logger import get_logger
from src.exceptions import LLMError
from config.settings import GROQ_API_KEY, LLM_MODEL

logger = get_logger("query_rewriter")

def rewrite_query(original_query: str) -> str:
    """Rewrite a vague query into a better search query"""
    try:
        client = Groq(api_key=GROQ_API_KEY)

        prompt = f"""You are a search query optimizer for a 
university syllabus RAG system.

Your job is to rewrite the user's question into a clear, 
specific search query that will find the most relevant 
information in a university syllabus document.

Rules:
- Make the query specific and detailed
- Keep it under 100 words
- Do NOT answer the question — only rewrite it
- Return ONLY the rewritten query, nothing else

Original question: {original_query}

Rewritten search query:"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )

        rewritten = response.choices[0].message.content.strip()
        logger.info(f"Original: '{original_query}'")
        logger.info(f"Rewritten: '{rewritten}'")
        return rewritten

    except Exception as e:
        logger.warning(
            f"Query rewriting failed, using original: {e}"
        )
        return original_query  # fallback to original


def rewrite_if_needed(query: str) -> str:
    """Only rewrite if query seems vague or too short"""

    # These signals mean the query needs rewriting
    needs_rewrite = (
        len(query.split()) < 5 or      # too short
        query.endswith("?") is False or # not a question
        len(query) < 20                 # too brief
    )

    if needs_rewrite:
        logger.info(f"Query needs rewriting: '{query}'")
        return rewrite_query(query)
    else:
        logger.info(f"Query is good, no rewrite needed")
        return query