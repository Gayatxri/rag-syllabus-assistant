from groq import Groq
from src.logger import get_logger
from src.exceptions import LLMError
from config.settings import GROQ_API_KEY, LLM_MODEL, MAX_TOKENS

logger = get_logger("generator")

def get_llm_client() -> Groq:
    """Initialize Groq client"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        logger.info(f"LLM client ready: {LLM_MODEL}")
        return client
    except Exception as e:
        raise LLMError(f"Failed to initialize LLM: {e}")

def build_prompt(question: str, chunks: list,
                 metadatas: list) -> str:
    """Build prompt with context and sources"""
    context_parts = []
    for i, (chunk, meta) in enumerate(
        zip(chunks, metadatas)
    ):
        source = meta.get("source", "unknown")
        page = meta.get("page", 0)
        context_parts.append(
            f"[Source: {source}, Page: {page}]\n{chunk}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful academic assistant.
Answer the question using ONLY the context below.
Always mention which document and page your answer comes from.
If the answer is not in the context, say 
"I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

def generate_answer(client: Groq, prompt: str) -> str:
    """Generate answer using LLM"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS
        )
        answer = response.choices[0].message.content
        logger.info("Answer generated successfully")
        return answer
    except Exception as e:
        raise LLMError(f"Answer generation failed: {e}")