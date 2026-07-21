"""
ChronoRAG — Streamlit Demo App
Temporal Document Trust Scoring for RAG Systems
"""

import streamlit as st
import math
from chronorag_engine import (
    init_engine, naive_retrieve, chronorag_retrieve,
    temporal_trust_score, CURRENT_YEAR
)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="ChronoRAG",
    page_icon="⏳",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #0e1117; }
.stApp { background-color: #0e1117; }
.metric-box {
    background: #1a1d27;
    border-radius: 10px;
    padding: 16px 20px;
    border: 1px solid #2e3250;
    margin-bottom: 8px;
}
.year-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
}
.badge-2020 { background:#3d1515; color:#f08080; }
.badge-2022 { background:#3d3415; color:#f0c060; }
.badge-2024 { background:#153d1d; color:#60d080; }
.correct   { color: #60d080; font-weight: 600; }
.incorrect { color: #f08080; font-weight: 600; }
.tts-bar-bg {
    background: #2e3250;
    border-radius: 6px;
    height: 10px;
    width: 100%;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Load engine (cached) ──────────────────────────────────────
@st.cache_resource
def load():
    return init_engine("data")

chunks, vec, mat, docs, tts_map = load()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/time-machine.png", width=60)
    st.title("ChronoRAG")
    st.caption("Temporal Trust Scoring for RAG Systems")
    st.divider()

    st.subheader("⚙️ Parameters")
    alpha = st.slider(
        "Temporal Weight (α)",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="0 = pure semantic | 1 = pure temporal"
    )
    lam = st.slider(
        "Decay Rate (λ)",
        min_value=0.1, max_value=0.8, value=0.3, step=0.05,
        help="Higher = faster knowledge decay"
    )
    st.divider()

    st.subheader("📄 Knowledge Base")
    for doc in docs:
        tts = temporal_trust_score(doc["year"], lam)
        age = CURRENT_YEAR - doc["year"]
        bar_width = int(tts * 100)
        color = "#60d080" if doc["year"] == 2024 else (
                "#f0c060" if doc["year"] == 2022 else "#f08080")
        st.markdown(f"""
        <div style='margin-bottom:10px'>
            <span style='font-size:13px;font-weight:500;color:#ccc'>
                {doc['filename'].replace('leave_policy_','v').replace('.md','')}
            </span>
            <span style='font-size:11px;color:#888;margin-left:6px'>
                age={age}yr · TTS={tts}
            </span>
            <div class='tts-bar-bg'>
                <div style='background:{color};height:10px;
                     border-radius:6px;width:{bar_width}%'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.caption("Formula: S = (1-α)·sim + α·TTS")
    st.caption(f"TTS(d) = e^(-{lam}·age)")

# ── Main area ─────────────────────────────────────────────────
st.title("⏳ ChronoRAG")
st.markdown(
    "**Does your RAG chatbot trust outdated documents?** "
    "Ask any HR policy question and see how Naive RAG retrieves "
    "old answers while ChronoRAG always finds the current one."
)
st.divider()

# Suggested questions
SUGGESTED = [
    "How many days of privilege leave do employees receive annually?",
    "How many weeks of maternity leave are female employees entitled to?",
    "What is the monthly WFH stipend for remote work?",
    "How many days of paternity leave does the company provide?",
    "How many days of sick leave are employees entitled to per year?",
    "When can an employee start working from home after joining?",
    "How many days can unused privilege leave be carried forward?",
    "How many work from home days per week are permitted?",
    "Is leave encashment permitted for employees?",
]

st.subheader("Try a question")
col_input, col_btn = st.columns([4, 1])
with col_input:
    query = st.text_input(
        "Your query",
        placeholder="e.g. How many days of privilege leave do employees get?",
        label_visibility="collapsed"
    )
with col_btn:
    search = st.button("🔍 Search", use_container_width=True)

st.markdown("**Or pick one:**")
cols = st.columns(3)
for i, q in enumerate(SUGGESTED):
    if cols[i % 3].button(q[:55] + "…" if len(q) > 55 else q,
                          key=f"q{i}", use_container_width=True):
        query  = q
        search = True

# ── Results ───────────────────────────────────────────────────
if search and query:
    st.divider()
    naive_results  = naive_retrieve(query, chunks, vec, mat, k=3)
    chrono_results = chronorag_retrieve(query, chunks, vec, mat,
                                        k=3, alpha=alpha, lam=lam)

    nt = naive_results[0]
    ct = chrono_results[0]

    # Summary banner
    naive_ok  = nt["year"] == 2024
    chrono_ok = ct["year"] == 2024

    bcolA, bcolB = st.columns(2)
    with bcolA:
        status = "✅ Retrieved current version" if naive_ok \
                 else "❌ Retrieved OUTDATED version"
        color  = "#153d1d" if naive_ok else "#3d1515"
        border = "#60d080" if naive_ok else "#f08080"
        st.markdown(f"""
        <div style='background:{color};border:1px solid {border};
             border-radius:10px;padding:14px 18px;text-align:center'>
            <div style='font-size:16px;font-weight:700;
                 color:{"#60d080" if naive_ok else "#f08080"}'>
                Naive RAG
            </div>
            <div style='font-size:13px;margin-top:4px;
                 color:{"#60d080" if naive_ok else "#f08080"}'>
                {status}
            </div>
            <div style='font-size:24px;margin-top:6px'>
                {"✅" if naive_ok else "❌"} Year: {nt["year"]}
            </div>
        </div>""", unsafe_allow_html=True)

    with bcolB:
        status = "✅ Retrieved current version" if chrono_ok \
                 else "❌ Retrieved OUTDATED version"
        color  = "#153d1d" if chrono_ok else "#3d1515"
        border = "#60d080" if chrono_ok else "#f08080"
        st.markdown(f"""
        <div style='background:{color};border:1px solid {border};
             border-radius:10px;padding:14px 18px;text-align:center'>
            <div style='font-size:16px;font-weight:700;
                 color:{"#60d080" if chrono_ok else "#f08080"}'>
                ChronoRAG (α={alpha}, λ={lam})
            </div>
            <div style='font-size:13px;margin-top:4px;
                 color:{"#60d080" if chrono_ok else "#f08080"}'>
                {status}
            </div>
            <div style='font-size:24px;margin-top:6px'>
                {"✅" if chrono_ok else "❌"} Year: {ct["year"]}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Side-by-side detailed results
    left, right = st.columns(2)

    def year_badge(year):
        cls = {2020: "badge-2020", 2022: "badge-2022", 2024: "badge-2024"}[year]
        return f"<span class='year-badge {cls}'>📄 {year} Policy</span>"

    def render_chunk(result, col, title, show_tts=False, show_final=False):
        with col:
            st.markdown(f"#### {title}")
            badge_html = year_badge(result["year"])
            tts_line   = (f"<br><b>TTS:</b> {result['tts']}"
                          if show_tts else "")
            final_line = (f"<br><b>Final Score:</b> {result['final_score']}"
                          if show_final else "")
            st.markdown(f"""
            <div class='metric-box'>
                {badge_html}
                <br>
                <b>Semantic Sim:</b> {result['sim']}
                {tts_line}
                {final_line}
            </div>""", unsafe_allow_html=True)

            with st.expander("📖 Retrieved chunk text", expanded=True):
                st.markdown(
                    f"<div style='font-size:13px;line-height:1.7;color:#ccc'>"
                    f"{result['text'][:500]}..."
                    f"</div>",
                    unsafe_allow_html=True
                )

            # All top-3
            st.markdown("**Top-3 retrieved (ranked):**")
            for i, r in enumerate(naive_results if not show_tts
                                  else chrono_results):
                yr_color = (
                    "#60d080" if r["year"] == 2024 else
                    "#f0c060" if r["year"] == 2022 else "#f08080"
                )
                score_label = (
                    f"sim={r['sim']} · tts={r['tts']} · final={r['final_score']}"
                    if show_tts else f"sim={r['sim']}"
                )
                st.markdown(
                    f"<div style='font-size:12px;margin-bottom:4px;"
                    f"padding:6px 10px;background:#1a1d27;"
                    f"border-radius:6px;border-left:3px solid {yr_color}'>"
                    f"#{i+1} · "
                    f"<span style='color:{yr_color};font-weight:600'>"
                    f"{r['year']}</span> · {score_label}"
                    f"</div>",
                    unsafe_allow_html=True
                )

    render_chunk(nt, left,  "🔴 Naive RAG",
                 show_tts=False, show_final=False)
    render_chunk(ct, right, "🟢 ChronoRAG",
                 show_tts=True,  show_final=True)

    # Score breakdown
    st.divider()
    st.subheader("🧮 Score Breakdown")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Naive RAG — Top Doc Year",   str(nt["year"]),
               delta="Outdated ❌" if nt["year"] < 2024 else "Current ✅")
    sc2.metric("ChronoRAG — Top Doc Year",   str(ct["year"]),
               delta="Current ✅" if ct["year"] == 2024 else "Outdated ❌")
    sc3.metric("TTS Advantage (2024 vs 2022)",
               f"{tts_map[2024] - tts_map.get(2022,0):.4f}",
               delta="Temporal bonus for current doc")

# ── About section ─────────────────────────────────────────────
with st.expander("ℹ️ What is ChronoRAG?"):
    st.markdown("""
**ChronoRAG** solves the *Temporal Retrieval Fallacy* — the tendency of
standard RAG systems to retrieve outdated document versions because
older documents often have higher semantic similarity to queries.

**The Formula:**
```
S_ChronoRAG = (1 - α) × semantic_similarity + α × TTS
TTS(document) = e^(-λ × age_in_years)
```

**Key findings from our experiment:**
- Naive RAG: **0/10** correct on versioned HR policy queries
- ChronoRAG: **10/10** correct with α ≥ 0.3

*Research paper: Janani S & Dhinakaran K, 2026*
    """)
