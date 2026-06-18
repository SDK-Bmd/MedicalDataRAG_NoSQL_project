import time

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------- Configuration ----------
QDRANT_URL = "http://localhost:6333"
COLLECTION = "mtsamples"
# Medical-domain model -- 768 dimensions, much better recall on clinical text
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO" #"all-MiniLM-L6-v2"
VECTOR_DIM = 768 #384
BATCH_SIZE = 64
CSV_PATH = "mtsamples.csv"
BATCH_SIZE = 32                       # smaller batches -> smaller payloads
PAYLOAD_TRANSCRIPTION_LIMIT = 2000
HTTP_TIMEOUT_SEC = 120                # generous client-side timeout
UPSERT_MAX_RETRIES = 3
INDEXING_THRESHOLD_DEFAULT = 20000    # 2000

def load_and_clean(path: str) -> pd.DataFrame:
    """Read the CSV and drop empty rows / normalize whitespace."""
    df = pd.read_csv(path)
    print(f"Raw rows: {len(df)}")

    df = df.dropna(subset=["transcription", "medical_specialty"])
    for col in ["medical_specialty", "sample_name", "description", "keywords"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    df = df.reset_index(drop=True)
    print(f"After cleaning: {len(df)} rows, "
          f"{df['medical_specialty'].nunique()} medical specialties")
    return df


def recreate_collection(client: QdrantClient) -> None:
    """Drop the collection if it exists, then create a fresh one
    with HNSW indexing temporarily disabled for fast bulk insert."""
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print(f"Deleted existing collection '{COLLECTION}'")

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        # disable indexing during bulk loading
        optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
    )

    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="medical_specialty",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print(f"Created collection '{COLLECTION}' "
          f"(dim={VECTOR_DIM}, cosine, indexing disabled)")


def enable_indexing(client: QdrantClient) -> None:
    """Re-enable HNSW indexing after the bulk insert finishes."""
    client.update_collection(
        collection_name=COLLECTION,
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=INDEXING_THRESHOLD_DEFAULT
        ),
    )
    print(f"Indexing re-enabled (threshold={INDEXING_THRESHOLD_DEFAULT}). "
          "Qdrant will now build the HNSW index in the background.")


def upsert_with_retry(client: QdrantClient, points: list) -> None:
    """Upsert a batch, retrying on transient failures."""
    delay = 2
    for attempt in range(1, UPSERT_MAX_RETRIES + 1):
        try:
            client.upsert(collection_name=COLLECTION, points=points, wait=True)
            return
        except Exception as exc:
            if attempt == UPSERT_MAX_RETRIES:
                raise
            print(f"  upsert attempt {attempt} failed ({type(exc).__name__}); "
                  f"retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2


def ingest(df: pd.DataFrame, client: QdrantClient,
           model: SentenceTransformer) -> None:
    """Embed and upsert in batches."""
    n = len(df)
    for start in tqdm(range(0, n, BATCH_SIZE), desc="Embedding + upserting"):
        end = min(start + BATCH_SIZE, n)
        batch = df.iloc[start:end]

        embeddings = model.encode(
            batch["transcription"].tolist(),
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=BATCH_SIZE,
        )

        points = [
            PointStruct(
                id=int(start + i),
                vector=embeddings[i].tolist(),
                payload={
                    "sample_name": row["sample_name"],
                    "medical_specialty": row["medical_specialty"],
                    "description": row["description"],
                    "keywords": row["keywords"],
                    "transcription": row["transcription"][:PAYLOAD_TRANSCRIPTION_LIMIT],
                },
            )
            for i, (_, row) in enumerate(batch.iterrows())
        ]

        upsert_with_retry(client, points)


def main() -> None:
    df = load_and_clean(CSV_PATH)

    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL, timeout=HTTP_TIMEOUT_SEC)
    recreate_collection(client)

    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    t0 = time.time()
    ingest(df, client, model)
    print(f"Ingestion finished in {time.time() - t0:.1f} s")

    enable_indexing(client)

    info = client.get_collection(COLLECTION)
    print(f"\nDone. Points in collection: {info.points_count}")
    print(f"Open the dashboard: {QDRANT_URL}/dashboard")


if __name__ == "__main__":
    main()