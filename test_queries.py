"""
query.py
--------
Three demonstration queries against the 'mtsamples' Qdrant collection:

  Q1. Pure semantic search (no filter) -- shows that matches are found even
      when the query does not share the same vocabulary as the documents.
  Q2. The same query, restricted to the 'Cardiovascular / Pulmonary'
      specialty via a payload filter (hybrid search).
  Q3. A different clinical theme, to show that the embedding model captures
      a wide range of medical concepts.

Run after ingest.py has populated the collection.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6333"
COLLECTION = "mtsamples"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def search(client: QdrantClient, model: SentenceTransformer,
           query_text: str, specialty: str | None = None,
           k: int = TOP_K):
    """Return the top-K most similar transcriptions, optionally filtered."""
    vector = model.encode(query_text).tolist()

    qfilter = None
    if specialty:
        qfilter = Filter(must=[FieldCondition(
            key="medical_specialty",
            match=MatchValue(value=specialty),
        )])

    response = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=qfilter,
        limit=k,
        with_payload=True,
    )
    return response.points


def print_results(query_text: str, specialty: str | None, points) -> None:
    header = f"Query: {query_text!r}"
    if specialty:
        header += f"  |  filter: medical_specialty == {specialty!r}"
    print("\n" + "=" * 78)
    print(header)
    print("=" * 78)

    if not points:
        print("(no results)")
        return

    for rank, p in enumerate(points, start=1):
        payload = p.payload or {}
        print(f"\n[{rank}] score={p.score:.4f}  "
              f"specialty={payload.get('medical_specialty', '?')}")
        print(f"    sample : {payload.get('sample_name', '')}")
        desc = payload.get("description", "")
        if desc:
            print(f"    desc   : {desc[:220]}")


def main() -> None:
    client = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(MODEL_NAME)

    # --- Q1: pure semantic search, cardiology theme expressed in lay terms
    q1 = "Patient with chest pain and shortness of breath when climbing stairs"
    print_results(q1, None, search(client, model, q1))

    # --- Q2: same query, filtered to cardiology specialty
    print_results(
        q1, "Cardiovascular / Pulmonary",
        search(client, model, q1, specialty="Cardiovascular / Pulmonary"),
    )

    # --- Q3: a different clinical theme
    q3 = "Severe recurrent headaches with visual disturbances and nausea"
    print_results(q3, None, search(client, model, q3))


if __name__ == "__main__":
    main()