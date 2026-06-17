import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, PayloadSchemaType,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------- Configuration ----------
QDRANT_URL = "http://localhost:6333"
COLLECTION = "mtsamples"
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384
CSV_PATH = "mtsamples.csv"
BATCH_SIZE = 64
PAYLOAD_TRANSCRIPTION_LIMIT = 2000   # keep payload reasonable

def load_and_clean(path: str) -> pd.DataFrame:
    """Read the CSV and drop empty rows / normalize whitespace."""
    df = pd.read_csv(path)
    print(f"Raw rows: {len(df)}")

    # The two columns we actually need
    df = df.dropna(subset=["transcription", "medical_specialty"])

    # Normalize text fields
    for col in ["medical_specialty", "sample_name", "description", "keywords"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    df = df.reset_index(drop=True)
    print(f"After cleaning: {len(df)} rows, "
          f"{df['medical_specialty'].nunique()} medical specialties")
    return df


def recreate_collection(client: QdrantClient) -> None:
    """Drop the collection if it exists, then create a fresh one."""
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print(f"Deleted existing collection '{COLLECTION}'")

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

    # Index the specialty field so filter queries are fast
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="medical_specialty",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print(f"Created collection '{COLLECTION}' (dim={VECTOR_DIM}, cosine)")


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

        client.upsert(collection_name=COLLECTION, points=points)


def main() -> None:
    df = load_and_clean(CSV_PATH)

    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)
    recreate_collection(client)

    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    ingest(df, client, model)

    info = client.get_collection(COLLECTION)
    print(f"\nDone. Points in collection: {info.points_count}")
    print(f"Open the dashboard: {QDRANT_URL}/dashboard")


if __name__ == "__main__":
    main()