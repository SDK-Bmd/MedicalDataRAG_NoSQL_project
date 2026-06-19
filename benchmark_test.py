import statistics
import time

from qdrant_client import QdrantClient
from qdrant_client.models import QuantizationSearchParams, SearchParams
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6333"
COLLECTION = "mtsamples"
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
K = 10
WARMUP = 2

QUERIES = [
    "Patient with chest pain and shortness of breath when climbing stairs",
    "Severe recurrent headaches with visual disturbances and nausea",
    "Knee surgery recovery and physical therapy",
    "Skin rash with itching after antibiotic treatment",
    "Diabetic patient with foot ulcer and neuropathy",
    "Sudden weakness on the right side and slurred speech",
    "Lower back pain radiating down the leg",
    "Persistent cough with weight loss and night sweats",
    "Abdominal pain after eating fatty foods",
    "Newborn with jaundice and poor feeding",
]


def run_query(client, vector, *, use_quantization: bool):
    params = SearchParams(
        quantization=QuantizationSearchParams(
            ignore=not use_quantization,   # True -> bypass quantization
            rescore=use_quantization,
            oversampling=2.0 if use_quantization else 1.0,
        )
    )
    t0 = time.perf_counter()
    points = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=K,
        with_payload=False,
        search_params=params,
    ).points
    elapsed_ms = (time.perf_counter() - t0) * 1000
    ids = [p.id for p in points]
    return ids, elapsed_ms


def main():
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    model = SentenceTransformer(MODEL_NAME)

    # Encode all queries up-front so the cost is not counted in latency
    print("Encoding queries...")
    vectors = [model.encode(q).tolist() for q in QUERIES]

    # Warmup
    for v in vectors[:WARMUP]:
        run_query(client, v, use_quantization=True)
        run_query(client, v, use_quantization=False)

    gt_latencies, qz_latencies, recalls = [], [], []

    for q, v in zip(QUERIES, vectors):
        gt_ids, gt_ms = run_query(client, v, use_quantization=False)
        qz_ids, qz_ms = run_query(client, v, use_quantization=True)

        overlap = len(set(gt_ids) & set(qz_ids))
        recall = overlap / K

        gt_latencies.append(gt_ms)
        qz_latencies.append(qz_ms)
        recalls.append(recall)

        print(f"\nQuery: {q[:60]}...")
        print(f"  ground truth (no quant): {gt_ms:6.1f} ms")
        print(f"  quantized + rescore   : {qz_ms:6.1f} ms")
        print(f"  recall@{K}             : {recall:.2f}")

    def summarise(label, values):
        mean = statistics.mean(values)
        p95 = sorted(values)[int(0.95 * len(values)) - 1]
        print(f"  {label:30s}  mean={mean:6.1f}  p95={p95:6.1f}")

    print("\n" + "=" * 60)
    print(f"Summary over {len(QUERIES)} queries (K={K})")
    print("=" * 60)
    summarise("Latency no-quant (ms)", gt_latencies)
    summarise("Latency quantized (ms)", qz_latencies)
    speedup = statistics.mean(gt_latencies) / statistics.mean(qz_latencies)
    print(f"  speedup (no-quant / quantized): {speedup:.2f}x")
    print(f"  mean recall@{K}: {statistics.mean(recalls):.3f}")


if __name__ == "__main__":
    main()