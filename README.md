# ⏳ ChronoRAG

**Temporal Document Trust Scoring for RAG Systems**

> *"When should a chatbot stop trusting its own knowledge base?"*

---

## 🔬 What is ChronoRAG?

ChronoRAG solves the **Temporal Retrieval Fallacy** — the systematic
tendency of standard RAG systems to retrieve outdated document versions
because older documents often have higher semantic similarity to queries
than newer ones.

### The Formula

```
S_ChronoRAG = (1 - α) × semantic_similarity + α × TTS(document)

TTS(document) = e^(-λ × age_in_years)
```

| Parameter | Role | Default |
|-----------|------|---------|
| α (alpha) | Temporal weight (0 = pure semantic, 1 = pure temporal) | 0.4 |
| λ (lambda) | Domain decay rate (higher = faster decay) | 0.3 |

---

## 📊 Key Results

| System | Correct Retrievals (10 queries) | Accuracy |
|--------|--------------------------------|----------|
| Naive RAG | 0 / 10 | 0% |
| ChronoRAG | 10 / 10 | 100% |

---

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/chronorag
cd chronorag
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
chronorag/
├── app.py                    # Streamlit demo app
├── chronorag_engine.py       # Core retrieval engine
├── requirements.txt
├── README.md
└── data/
    ├── leave_policy_2020.md  # Outdated policy (2020)
    ├── leave_policy_2022.md  # Intermediate policy (2022)
    └── leave_policy_2024.md  # Current policy (2024)
```

---

## 📄 Research Paper

*ChronoRAG: Temporal Document Trust Scoring for Retrieval-Augmented
Generation — When Should a Chatbot Stop Trusting Its Own Knowledge Base?*

Janani S & Dhinakaran K
Dhanalakshmi College of Engineering, Chennai, India, 2026

---

## 🔧 Built With

- Python · Streamlit · scikit-learn · TF-IDF · Cosine Similarity
