from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Tuple

# Load reranker model and tokenizer (only once)
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def rerank_chunks(query: str, chunks: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Reranks the given chunks based on relevance to the query using a cross-encoder reranker model.

    Args:
        query (str): The input query.
        chunks (List[str]): Candidate text chunks.
        top_k (int): Number of top reranked results to return.

    Returns:
        List[Tuple[str, float]]: Top-K chunks and their scores (in descending order).
    """
    # Prepare query-chunk pairs
    pairs = [(query, chunk) for chunk in chunks]

    # Tokenize pairs
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        scores = model(**inputs).logits.squeeze().tolist()

    if isinstance(scores, float):  # when only one chunk
        scores = [scores]

    # Pair chunks with scores and sort
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return scored_chunks[:top_k]



if __name__ == "__main__":
    query = "What is the capital of France?"
    chunks = [
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital of Germany.",
        "France is a country in Western Europe.",
        "Madrid is the capital of Spain."
    ]

    top_k_results = rerank_chunks(query, chunks, top_k=2)
    for chunk, score in top_k_results:
        print(f"{score:.4f} - {chunk}")
