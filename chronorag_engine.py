"""
ChronoRAG Engine
Temporal Trust Scoring for RAG Systems
"""
import math
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CURRENT_YEAR = 2024

# ── Document loading ──────────────────────────────────────────
def load_documents(data_dir="data"):
    docs = []
    version_map = {
        "leave_policy_2020.md": 2020,
        "leave_policy_2022.md": 2022,
        "leave_policy_2024.md": 2024,
    }
    for fname, year in version_map.items():
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append({"filename": fname, "year": year, "content": content})
    return sorted(docs, key=lambda x: x["year"])

# ── Chunking ──────────────────────────────────────────────────
def chunk_documents(docs, chunk_size=400, overlap=60):
    all_chunks = []
    for doc in docs:
        text, start = doc["content"], 0
        while start < len(text):
            end   = min(start + chunk_size, len(text))
            chunk = text[start:end]
            all_chunks.append({
                "text":     chunk,
                "year":     doc["year"],
                "filename": doc["filename"],
            })
            start += chunk_size - overlap
    return all_chunks

# ── Temporal Trust Score ──────────────────────────────────────
def temporal_trust_score(year, lam=0.3):
    """TTS(d) = exp(-lambda * age_in_years)"""
    age = CURRENT_YEAR - year
    return round(math.exp(-lam * age), 4)

# ── Build TF-IDF index ────────────────────────────────────────
def build_index(chunks):
    texts      = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    matrix     = vectorizer.fit_transform(texts)
    return vectorizer, matrix

# ── Retrieval ─────────────────────────────────────────────────
def naive_retrieve(query, chunks, vectorizer, matrix, k=3):
    qv   = vectorizer.transform([query])
    sims = cosine_similarity(qv, matrix)[0]
    idxs = np.argsort(sims)[::-1][:k]
    results = []
    for i in idxs:
        c = chunks[i].copy()
        c["sim"]         = round(float(sims[i]), 4)
        c["tts"]         = temporal_trust_score(c["year"])
        c["final_score"] = c["sim"]
        c["method"]      = "Naive RAG"
        results.append(c)
    return results

def chronorag_retrieve(query, chunks, vectorizer, matrix, k=3, alpha=0.4, lam=0.3):
    qv   = vectorizer.transform([query])
    sims = cosine_similarity(qv, matrix)[0]
    candidates = []
    for c, sim in zip(chunks, sims):
        t     = temporal_trust_score(c["year"], lam)
        final = (1 - alpha) * float(sim) + alpha * t
        ch    = c.copy()
        ch["sim"]         = round(float(sim), 4)
        ch["tts"]         = t
        ch["final_score"] = round(final, 4)
        ch["method"]      = "ChronoRAG"
        candidates.append(ch)
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates[:k]

# ── Initialise (call once) ────────────────────────────────────
def init_engine(data_dir="data"):
    docs   = load_documents(data_dir)
    chunks = chunk_documents(docs)
    vec, mat = build_index(chunks)
    tts_map = {doc["year"]: temporal_trust_score(doc["year"]) for doc in docs}
    return chunks, vec, mat, docs, tts_map
