import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from src.logger import get_logger
from src.query_rewriter import rewrite_query, rewrite_if_needed

logger = get_logger("main")

def main():
    logger.info("="*50)
    logger.info("Day 11 — Query Rewriting Tests")
    logger.info("="*50)

    # Test queries — bad to good
    test_queries = [
        "tell me unit 1",
        "exam?",
        "disasters",
        "what is the sendai framework for disaster risk?",
        "credits",
        "types of hazards in the syllabus",
    ]

    print("\n" + "="*55)
    print("🔄 Query Rewriting Test")
    print("="*55)

    for query in test_queries:
        print(f"\n❓ Original:  '{query}'")
        rewritten = rewrite_if_needed(query)
        if rewritten != query:
            print(f"✅ Rewritten: '{rewritten}'")
        else:
            print(f"✅ No rewrite needed — query is good!")
        print("-"*55)

if __name__ == "__main__":
    main()