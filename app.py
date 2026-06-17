"""
app.py
------
Streamlit interface for semantic search over the MTSamples Qdrant collection.

Features:
  - Free-text query, encoded on the fly with sentence-transformers
  - Sidebar filter: medical specialty (multi-select, payload-indexed)
  - Sliders for the number of results (top-K) and a minimum similarity score
  - One-click example queries for the demo
  - Per-result card with score, specialty, description, and expandable
    transcription

Run:
    streamlit run app.py

Prerequisites:
  - Qdrant running locally (docker compose up -d)
  - Collection 'mtsamples' already populated (python data-ingest.py)
"""

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from sentence_transformers import SentenceTransformer

# ---------- Configuration ----------
QDRANT_URL = "http://localhost:6333"
COLLECTION = "mtsamples"
MODEL_NAME = "all-MiniLM-L6-v2"

EXAMPLE_QUERIES = [
    "Patient with chest pain and shortness of breath when climbing stairs",
    "Severe recurrent headaches with visual disturbances and nausea",
    "Knee surgery recovery and physical therapy",
    "Skin rash with itching after antibiotic treatment",
]

# ---------- Page setup ----------
st.set_page_config(
    page_title="MTSamples Semantic Search",
    page_icon="🩺",
    layout="wide",
)


# ---------- Cached resources ----------
@st.cache_resource
def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


@st.cache_resource
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@st.cache_data(ttl=600)
def get_collection_info(_client: QdrantClient) -> dict:
    info = _client.get_collection(COLLECTION)
    return {
        "points": info.points_count,
        "dim": info.config.params.vectors.size,
    }


@st.cache_data(ttl=600)
def get_specialties(_client: QdrantClient) -> list[str]:
    """Scroll through all points to collect the unique specialties."""
    specialties: set[str] = set()
    offset = None
    while True:
        points, offset = _client.scroll(
            collection_name=COLLECTION,
            limit=500,
            offset=offset,
            with_payload=["medical_specialty"],
            with_vectors=False,
        )
        for p in points:
            value = (p.payload or {}).get("medical_specialty")
            if value:
                specialties.add(value)
        if offset is None:
            break
    return sorted(specialties)


# ---------- Connect ----------
try:
    client = get_client()
    info = get_collection_info(client)
    specialties = get_specialties(client)
except Exception as exc:
    st.error(f"Could not connect to Qdrant at {QDRANT_URL}.\n\n{exc}")
    st.info("Make sure the container is running: `docker compose up -d`, "
            "and that ingest.py has been executed.")
    st.stop()


# ---------- Sidebar ----------
st.sidebar.title("🩺 MTSamples Search")
st.sidebar.success(
    f"Connected · {info['points']:,} documents · {info['dim']}-dim vectors"
)

st.sidebar.markdown("### Filters")

selected_specialties = st.sidebar.multiselect(
    "Medical specialty",
    options=specialties,
    default=[],
    help="Leave empty to search across all specialties.",
)

top_k = st.sidebar.slider("Number of results (top-K)", 1, 20, 5)
min_score = st.sidebar.slider("Minimum similarity score", 0.0, 1.0, 0.0, 0.05)

with st.sidebar.expander("About"):
    st.markdown(
        "Vectors are produced by **all-MiniLM-L6-v2** (384 dimensions) "
        "and indexed in **Qdrant** with cosine distance.\n\n"
        f"Dashboard: [open]({QDRANT_URL}/dashboard)"
    )


# ---------- Main area ----------
st.title("Semantic search over clinical transcriptions")
st.caption(
    "Type a description in plain English — the engine retrieves the closest "
    "real clinical notes even when they use different vocabulary."
)


# Initialise session state for the query box
if "query_input" not in st.session_state:
    st.session_state.query_input = ""


def _set_query(value: str) -> None:
    st.session_state.query_input = value


with st.expander("Example queries", expanded=False):
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUERIES):
        cols[i % 2].button(
            q, key=f"ex_{i}", on_click=_set_query, args=(q,),
            use_container_width=True,
        )

query = st.text_area(
    "Query",
    height=90,
    key="query_input",
    placeholder="e.g. patient with sudden weakness on the right side and slurred speech",
)

go = st.button("Search", type="primary", disabled=not query.strip())


# ---------- Search & results ----------
if go:
    model = get_model()

    with st.spinner("Embedding query and searching Qdrant..."):
        vector = model.encode(query).tolist()

        qfilter = None
        if selected_specialties:
            qfilter = Filter(must=[FieldCondition(
                key="medical_specialty",
                match=MatchAny(any=selected_specialties),
            )])

        points = client.query_points(
            collection_name=COLLECTION,
            query=vector,
            query_filter=qfilter,
            limit=top_k,
            with_payload=True,
        ).points

    points = [p for p in points if p.score >= min_score]

    st.markdown(f"### {len(points)} result(s)")

    if not points:
        st.info(
            "No matching documents. Try lowering the minimum score, "
            "removing specialty filters, or rephrasing the query."
        )

    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        specialty = payload.get("medical_specialty", "?")
        sample = payload.get("sample_name") or "(untitled)"
        description = payload.get("description", "")
        keywords = payload.get("keywords", "")
        transcription = payload.get("transcription", "")

        with st.container(border=True):
            left, right = st.columns([1, 5])
            left.metric(label=f"#{rank}", value=f"{point.score:.3f}",
                        help="Cosine similarity (higher = closer)")
            right.markdown(f"**{sample}**")
            right.caption(f"Specialty: `{specialty}`")

            if description:
                st.markdown(f"_{description}_")
            if keywords:
                st.caption(f"**Keywords:** {keywords}")
            if transcription:
                with st.expander("Show transcription excerpt"):
                    st.text(transcription)