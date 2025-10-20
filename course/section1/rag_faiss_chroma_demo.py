#!/usr/bin/env python3
"""
RAG Vector Search Demo: FAISS & Chroma
--------------------------------------
A single-file, runnable guide that:
- builds & queries FAISS (exact / HNSW) and Chroma (HNSW) indexes
- prints explanations, parameter choices, and results as Markdown
- shows how to persist & reload indexes

Usage (pick one):
  python rag_faiss_chroma_demo.py --faiss exact
  python rag_faiss_chroma_demo.py --faiss hnsw
  python rag_faiss_chroma_demo.py --chroma

Requirements:
  pip install sentence-transformers faiss-cpu chromadb

Model download tip:
- To avoid re-downloading the model across runs/containers:
    export HF_HOME=/models/hf
    # or use snapshot_download(...) to bake the model into an image/volume
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

# ---- Embeddings ----
def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load an embedding model."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    # For cosine similarity, we will normalize embeddings.
    return model

def embed_texts(model, texts: List[str], normalize: bool = True):
    """Return numpy array of embeddings."""
    import numpy as np
    X = model.encode(texts, normalize_embeddings=normalize)
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    return X

# ---- Demo corpus ----
DOCS = [
    "The Houston Rockets play at Toyota Center in Houston, Texas.",
    "NBA fantasy strategy often favors players with high usage and pace.",
    "Chroma is a simple local vector database often used for RAG demos.",
    "FAISS is a fast vector similarity library created by Facebook AI Research.",
    "Elasticsearch supports hybrid search with BM25 and dense vectors.",
    "Milvus and Weaviate are open-source vector databases for large-scale search."
]
IDS = [f"doc-{i+1}" for i in range(len(DOCS))]
METAS = [
    {"source": "sports"}, {"source": "nba"}, {"source": "rag"},
    {"source": "faiss"}, {"source": "elastic"}, {"source": "oss"}
]

QUESTION = "Where do the Houston Rockets play?"

# ---- Markdown helpers ----
def md_h1(text): print(f"# {text}")
def md_h2(text): print(f"## {text}")
def md_h3(text): print(f"### {text}")
def md_p(text=""): print(text)
def md_code(code, lang=""): print(f"```{lang}\n{code}\n```")
def md_table(headers: List[str], rows: List[List[str]]):
    head = " | ".join(headers)
    sep = " | ".join(["---"] * len(headers))
    print(head); print(sep)
    for r in rows: print(" | ".join(map(str, r)))

# ---- FAISS: Exact & HNSW ----
def faiss_exact_demo(model):
    """
    Exact search with cosine similarity via Inner Product on normalized vectors.
    Suitable for <= ~100k vectors or as a baseline.
    """
    import faiss
    import numpy as np

    md_h2("FAISS (Exact) — IndexFlatIP")
    md_p("- **Metric:** Cosine (via normalized vectors + Inner Product)")
    md_p("- **Why:** Simple, no training; great baseline and small corpora.")
    md_p()

    X = embed_texts(model, DOCS, normalize=True)
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)

    # Persist
    faiss.write_index(index, "faiss_exact.index")
    Path("faiss_meta.json").write_text(json.dumps({"ids": IDS, "docs": DOCS, "metas": METAS}))

    # Query
    qv = embed_texts(model, [QUESTION], normalize=True)
    scores, idxs = index.search(qv, k=3)

    md_h3("Query")
    md_code(QUESTION)
    md_h3("Top-3 Results (doc, score, meta)")
    rows = []
    for rank, i in enumerate(idxs[0], 1):
        rows.append([rank, IDS[i], f"{scores[0][rank-1]:.4f}", DOCS[i], METAS[i]])
    md_table(["Rank", "ID", "Score", "Text", "Meta"], rows)

def faiss_hnsw_demo(model, M=32, efConstruction=200, efSearch=80):
    """
    HNSW ANN: Good default for medium/large corpora.
    Trade-offs: higher efSearch = better recall, slower queries.
    """
    import faiss
    import numpy as np

    md_h2("FAISS (HNSW) — IndexHNSWFlat")
    md_p("- **Params:**")
    md_table(["Name", "What it does", "Good starting value"],
             [["M", "Graph out-degree (connectivity)", str(M)],
              ["efConstruction", "Build-time accuracy/speed", str(efConstruction)],
              ["efSearch", "Query-time recall/latency", str(efSearch)]])
    md_p()

    X = embed_texts(model, DOCS, normalize=True)
    d = X.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch
    index.add(X)

    faiss.write_index(index, "faiss_hnsw.index")
    Path("faiss_meta.json").write_text(json.dumps({"ids": IDS, "docs": DOCS, "metas": METAS}))

    qv = embed_texts(model, [QUESTION], normalize=True)
    scores, idxs = index.search(qv, k=3)

    md_h3("Query")
    md_code(QUESTION)
    md_h3("Top-3 Results (doc, score, meta)")
    rows = []
    for rank, i in enumerate(idxs[0], 1):
        rows.append([rank, IDS[i], f"{scores[0][rank-1]:.4f}", DOCS[i], METAS[i]])
    md_table(["Rank", "ID", "Score", "Text", "Meta"], rows)

# ---- Chroma ----
def chroma_demo(hnsw_space="cosine", M=32, efConstruction=200, ef=80, persist_path="./chroma_db_demo"):
    """
    Chroma using HNSW under the hood with persistence & metadata filtering.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    md_h2("Chroma (HNSW) — PersistentClient")
    md_p("- **Distance/space:** cosine (or 'l2', 'ip')")
    md_p("- **HNSW params:** M, efConstruction, ef (query)")
    md_p("- **Persistence:** saves to a folder; survives restarts.")
    md_p()

    client = chromadb.PersistentClient(path=persist_path)

    # Provide a local embedding function (Chroma can also run in server mode)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def embed_fn(texts: List[str]):
        return model.encode(texts, normalize_embeddings=True).tolist()

    col = client.get_or_create_collection(
        name="docs",
        metadata={
            "hnsw:space": hnsw_space,
            "hnsw:M": M,
            "hnsw:efConstruction": efConstruction,
            "hnsw:ef": ef
        },
        embedding_function=embed_fn
    )

    # Upsert (id is required). Upserts are idempotent by id.
    col.upsert(documents=DOCS, ids=IDS, metadatas=METAS)

    # Query
    res = col.query(query_texts=[QUESTION], n_results=3, include=["documents", "metadatas", "distances"])

    md_h3("Query")
    md_code(QUESTION)
    md_h3("Top-3 Results (document, distance, meta)")
    rows = []
    docs = res["documents"][0]
    dists = res["distances"][0]
    metas = res["metadatas"][0]
    ids = res.get("ids", [["?","?","?"]])[0]
    for rank, (doc, dist, meta, _id) in enumerate(zip(docs, dists, metas, ids), 1):
        rows.append([rank, _id, f"{dist:.4f}", doc, meta])
    md_table(["Rank", "ID", "Distance↓ (cosine)", "Text", "Meta"], rows)

    # Server-side metadata filter example
    md_h3("Filter Example (where={'source': 'sports'})")
    res2 = col.query(
        query_texts=[QUESTION],
        n_results=3,
        where={"source": "sports"},
        include=["documents", "metadatas", "distances"]
    )
    docs2 = res2["documents"][0]
    md_p(f"- Returned {len(docs2)} doc(s): {docs2!r}")

# ---- Guidance table ----
def guidance_md():
    md_h2("Parameter Decisions — Quick Guide")
    md_table(
        ["Layer", "Knob", "Why it matters", "Starting point"],
        [
            ["Embeddings", "Chunk size", "Context coherence vs coverage", "200–500 tokens"],
            ["Embeddings", "Metric", "Cosine robust to scale; L2 for Euclidean", "Cosine + normalize"],
            ["Retrieval", "k (top-k)", "Recall ↑ with k; re-rank later", "8–10"],
            ["FAISS HNSW", "M", "Graph connectivity → recall/ram", "32"],
            ["FAISS HNSW", "efSearch", "Recall vs latency at query time", "80"],
            ["FAISS IVF", "nlist", "How many clusters → recall/speed", "4096 (for millions)"],
            ["FAISS IVF", "nprobe", "Clusters probed at query time", "16"],
            ["Chroma", "hnsw:ef", "Query recall vs latency", "80"],
            ["Chroma", "where filter", "Server-side narrowing", "Use tags (source, date)"],
        ]
    )
    md_p()

# ---- Main ----
def main():
    parser = argparse.ArgumentParser(description="RAG Vector Search Demo: FAISS & Chroma")
    parser.add_argument("--faiss", choices=["exact", "hnsw"], help="Run a FAISS demo (exact or hnsw)")
    parser.add_argument("--chroma", action="store_true", help="Run a Chroma demo")
    args = parser.parse_args()

    md_h1("RAG Vector Search Demo: FAISS & Chroma")
    md_p("This guide builds a tiny corpus, indexes it with FAISS and/or Chroma, then queries with a question.")
    md_p("Outputs below are **live results** (they may differ slightly by environment), printed as Markdown.")

    md_h2("Setup")
    md_code(
        "pip install sentence-transformers faiss-cpu chromadb\n"
        "# Optional: persist HF cache so the model isn't re-downloaded\n"
        "export HF_HOME=/models/hf",
        lang="bash"
    )

    md_h2("Corpus")
    rows = [[i+1, IDS[i], DOCS[i], METAS[i]] for i in range(len(DOCS))]
    md_table(["#", "ID", "Text", "Meta"], rows)

    # Load embedder once
    model = get_embedder()

    # Run requested demos
    if args.faiss == "exact":
        faiss_exact_demo(model)

    if args.faiss == "hnsw":
        faiss_hnsw_demo(model)

    if args.chroma:
        chroma_demo()

    guidance_md()

    md_h2("Expected Output Shape (Example)")
    md_p("Your scores/distances will vary, but the tables will look like this:")
    md_code(
        """## FAISS (Exact) — IndexFlatIP
### Query
Where do the Houston Rockets play?

### Top-3 Results (doc, score, meta)
Rank | ID | Score | Text | Meta
--- | --- | --- | --- | ---
1 | doc-1 | 0.78 | The Houston Rockets play at Toyota Center in Houston, Texas. | {'source': 'sports'}
2 | doc-2 | 0.22 | NBA fantasy strategy often favors players with high usage and pace. | {'source': 'nba'}
3 | doc-3 | 0.11 | Chroma is a simple local vector database often used for RAG demos. | {'source': 'rag'}
""", lang="markdown"
    )

if __name__ == "__main__":
    main()
