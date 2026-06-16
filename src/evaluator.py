from groq import Groq
from src.logger import get_logger
from config.settings import GROQ_API_KEY, LLM_MODEL

logger = get_logger("evaluator")

def evaluate_faithfulness(client: Groq, question: str,
                          answer: str, context: str) -> float:
    """
    Faithfulness: Is the answer grounded in the context?
    Score 0.0 to 1.0
    """
    prompt = f"""Evaluate if the answer is faithful to 
the given context.

Question: {question}
Context: {context[:500]}
Answer: {answer[:300]}

Faithfulness means: every claim in the answer must be 
supported by the context. No made-up information.

Score from 0.0 to 1.0:
0.0 = answer contradicts or ignores context
0.5 = answer partially uses context
1.0 = answer is completely grounded in context

Reply with ONLY a number between 0.0 and 1.0."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        score = float(
            response.choices[0].message.content.strip().split()[0]
        )
        return max(0.0, min(1.0, score))
    except:
        return 0.5


def evaluate_answer_relevance(client: Groq, question: str,
                               answer: str) -> float:
    """
    Answer Relevance: Does the answer address the question?
    Score 0.0 to 1.0
    """
    prompt = f"""Evaluate if the answer is relevant 
to the question.

Question: {question}
Answer: {answer[:300]}

Relevance means: the answer directly addresses what 
was asked. Not off-topic or too vague.

Score from 0.0 to 1.0:
0.0 = completely off-topic
0.5 = partially answers the question
1.0 = directly and completely answers the question

Reply with ONLY a number between 0.0 and 1.0."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        score = float(
            response.choices[0].message.content.strip().split()[0]
        )
        return max(0.0, min(1.0, score))
    except:
        return 0.5


def evaluate_context_precision(client: Groq, question: str,
                                context: str) -> float:
    """
    Context Precision: Are retrieved chunks relevant?
    Score 0.0 to 1.0
    """
    prompt = f"""Evaluate if the retrieved context is 
relevant for answering the question.

Question: {question}
Retrieved Context: {context[:500]}

Context Precision means: the retrieved chunks should 
contain information useful for answering the question.

Score from 0.0 to 1.0:
0.0 = context is completely irrelevant
0.5 = context is partially relevant
1.0 = context is perfectly relevant

Reply with ONLY a number between 0.0 and 1.0."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        score = float(
            response.choices[0].message.content.strip().split()[0]
        )
        return max(0.0, min(1.0, score))
    except:
        return 0.5


def evaluate_single(question: str, answer: str,
                    context: str) -> dict:
    """Run all evaluations for one Q&A pair"""
    client = Groq(api_key=GROQ_API_KEY)

    faithfulness = evaluate_faithfulness(
        client, question, answer, context
    )
    answer_relevance = evaluate_answer_relevance(
        client, question, answer
    )
    context_precision = evaluate_context_precision(
        client, question, context
    )

    scores = {
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_precision": context_precision,
        "overall": round(
            (faithfulness + answer_relevance + context_precision) / 3,
            2
        )
    }

    logger.info(f"Evaluation scores: {scores}")
    return scores