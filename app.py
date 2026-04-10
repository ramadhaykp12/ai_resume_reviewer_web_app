"""
AI Resume Reviewer — Streamlit Interface
Menggunakan LangChain + Gemini untuk analisis kecocokan CV dengan Job Description
"""

import sys
import os

# Pastikan folder project (tempat app.py berada) ada di sys.path
# sehingga folder `agent/` selalu ditemukan sebagai module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from resume_agent import ResumeReviewAgent, ReviewResult, load_pdf_text

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Reviewer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background: #f5f4f0;
    color: #1a1a1a;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1280px; margin: 0 auto; }

/* ── Header ── */
.app-header {
    background: #1a1a1a;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    position: relative;
    overflow: hidden;
}
.app-header::after {
    content: 'CV';
    position: absolute;
    right: 3rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 8rem;
    font-weight: 800;
    color: rgba(255,255,255,0.04);
    letter-spacing: -0.05em;
    pointer-events: none;
}
.header-badge {
    background: #d4f460;
    color: #1a1a1a;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 5px 12px;
    border-radius: 99px;
    display: inline-block;
    margin-bottom: 0.6rem;
}
.header-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    line-height: 1.1;
    letter-spacing: -0.03em;
}
.header-sub {
    font-size: 0.9rem;
    color: #888;
    margin: 0.4rem 0 0;
    font-weight: 400;
}

/* ── Input Panels ── */
.panel {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1.5px solid #e8e6e0;
    height: 100%;
}
.panel-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Streamlit widget overrides ── */
[data-testid="stFileUploader"] {
    background: #f9f8f5 !important;
    border: 2px dashed #d0cdc5 !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: #1a1a1a !important; }

.stTextArea textarea {
    background: #f9f8f5 !important;
    border: 1.5px solid #e0ddd5 !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.875rem !important;
    color: #1a1a1a !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: #1a1a1a !important;
    box-shadow: 0 0 0 3px rgba(26,26,26,0.08) !important;
}

/* ── Analyze Button ── */
.stButton > button {
    background: #1a1a1a !important;
    color: #d4f460 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #2d2d2d !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(26,26,26,0.2) !important;
}

/* ── Result: Score Card ── */
.score-card {
    background: #1a1a1a;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    color: white;
    position: relative;
    overflow: hidden;
}
.score-card::before {
    content: '';
    position: absolute;
    bottom: -40px; right: -40px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(212,244,96,0.08);
}
.score-big {
    font-size: 5.5rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.04em;
}
.score-unit { font-size: 2.5rem; color: #888; }
.score-tag {
    display: inline-block;
    background: #d4f460;
    color: #1a1a1a;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 5px 14px;
    border-radius: 99px;
    margin-top: 0.75rem;
}
.score-bar-wrap {
    background: #2d2d2d;
    border-radius: 99px;
    height: 8px;
    margin-top: 1.2rem;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 1s ease;
}

/* ── Summary & Recommendation ── */
.summary-card {
    background: #fff;
    border: 1.5px solid #e8e6e0;
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
    line-height: 1.75;
    color: #444;
}
.rec-card {
    background: #f0fce8;
    border: 1.5px solid #c5e89a;
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    font-size: 0.9rem;
    line-height: 1.75;
    color: #3a5c1a;
}

/* ── Skill Chips ── */
.chip-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.5rem; }
.chip {
    font-size: 0.78rem;
    font-weight: 600;
    border-radius: 8px;
    padding: 4px 12px;
}
.chip-green { background: #e8fbd0; color: #2d6a0a; border: 1px solid #b8e88a; }
.chip-red   { background: #fde8e8; color: #8a0a0a; border: 1px solid #f0a8a8; }

/* ── Strengths / Gaps list ── */
.result-section {
    background: #fff;
    border: 1.5px solid #e8e6e0;
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
}
.section-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.list-item {
    padding: 0.7rem 0;
    border-bottom: 1px solid #f0ede8;
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
}
.list-item:last-child { border-bottom: none; padding-bottom: 0; }
.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-top: 6px;
    flex-shrink: 0;
}
.item-point { font-size: 0.87rem; font-weight: 600; color: #1a1a1a; }
.item-detail { font-size: 0.81rem; color: #777; margin-top: 2px; line-height: 1.5; }

/* ── Divider ── */
hr { border-color: #e8e6e0 !important; margin: 1.5rem 0 !important; }

/* ── Step indicators ── */
.step-row {
    display: flex;
    gap: 0.75rem;
    align-items: center;
    margin-bottom: 1.5rem;
}
.step-pill {
    background: #1a1a1a;
    color: #d4f460;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 99px;
    letter-spacing: 0.1em;
}
.step-text { font-size: 0.8rem; color: #888; font-weight: 500; }

/* Columns */
[data-testid="column"] { padding: 0 0.4rem !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def score_color(pct: int) -> str:
    if pct >= 80: return "#4caf50"
    if pct >= 60: return "#d4f460"
    if pct >= 40: return "#ff9800"
    return "#f44336"

def verdict_emoji(v: str) -> str:
    return {"Sangat Cocok": "🏆", "Cocok": "✅", "Cukup Cocok": "🔶",
            "Kurang Cocok": "⚠️", "Tidak Cocok": "❌"}.get(v, "📊")

@st.cache_resource
def get_agent() -> ResumeReviewAgent:
    """Inisialisasi agent sekali, di-cache oleh Streamlit."""
    return ResumeReviewAgent()


# ─── UI ────────────────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="app-header">
        <div>
            <div class="header-badge">⚡ Powered by LangChain + Gemini</div>
            <h1 class="header-title">AI Resume Reviewer</h1>
            <p class="header-sub">Analisis mendalam kecocokan CV dengan Job Description secara otomatis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_inputs():
    """Render area input PDF dan JD, kembalikan (file_bytes, jd_text)."""
    st.markdown("""
    <div class="step-row">
        <span class="step-pill">LANGKAH 1</span>
        <span class="step-text">Upload resume & masukkan job description</span>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">📄 Resume / CV (PDF)</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.markdown(
                f"<p style='font-size:0.78rem;color:#4caf50;margin:0.5rem 0 0;font-weight:600;'>"
                f"✓ {uploaded.name}</p>",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">💼 Job Description</div>', unsafe_allow_html=True)
        jd = st.text_area(
            "Job Description",
            placeholder="Tempelkan job description di sini...\n\nContoh:\nWe are looking for a Backend Engineer with 3+ years Python experience, familiar with FastAPI, PostgreSQL, Docker...",
            height=200,
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    return uploaded, jd


def render_analyze_button() -> bool:
    st.markdown("""
    <div class="step-row" style="margin-top:1rem;">
        <span class="step-pill">LANGKAH 2</span>
        <span class="step-text">Jalankan analisis AI</span>
    </div>
    """, unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        return st.button("🔍 Analisis Sekarang", use_container_width=True)


def render_results(result: ReviewResult):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-row">
        <span class="step-pill">HASIL</span>
        <span class="step-text">Laporan analisis kecocokan</span>
    </div>
    """, unsafe_allow_html=True)

    pct = result.match_percentage
    color = score_color(pct)
    emoji = verdict_emoji(result.verdict)

    # ── Row 1: Score + Summary/Rec ──
    col_score, col_info = st.columns([1, 2], gap="medium")

    with col_score:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-big">{pct}<span class="score-unit">%</span></div>
            <div class="score-tag">{emoji} {result.verdict}</div>
            <div class="score-bar-wrap">
                <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        if result.summary:
            st.markdown(
                f'<div class="summary-card">💬 {result.summary}</div>',
                unsafe_allow_html=True,
            )
        if result.recommendation:
            st.markdown(
                f'<div class="rec-card">💡 <strong>Rekomendasi:</strong> {result.recommendation}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Row 2: Skill chips ──
    col_matched, col_missing = st.columns(2, gap="medium")

    with col_matched:
        st.markdown(f"""
        <div class="result-section">
            <div class="section-title" style="color:#2d6a0a;">✅ Skill Dimiliki ({len(result.key_skills_matched)})</div>
            <div class="chip-wrap">
                {"".join(f'<span class="chip chip-green">{s}</span>' for s in result.key_skills_matched) or "<span style='color:#aaa;font-size:0.82rem;'>—</span>"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_missing:
        st.markdown(f"""
        <div class="result-section">
            <div class="section-title" style="color:#8a0a0a;">❌ Skill Dibutuhkan ({len(result.key_skills_missing)})</div>
            <div class="chip-wrap">
                {"".join(f'<span class="chip chip-red">{s}</span>' for s in result.key_skills_missing) or "<span style='color:#aaa;font-size:0.82rem;'>—</span>"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # ── Row 3: Strengths & Gaps ──
    col_str, col_gap = st.columns(2, gap="medium")

    with col_str:
        items_html = ""
        for s in result.strengths:
            items_html += f"""
            <div class="list-item">
                <span class="dot" style="background:#4caf50;"></span>
                <div>
                    <div class="item-point">{s.point}</div>
                    <div class="item-detail">{s.detail}</div>
                </div>
            </div>"""
        st.markdown(f"""
        <div class="result-section">
            <div class="section-title" style="color:#1a5c2a;">🌟 Kekuatan Kandidat ({len(result.strengths)})</div>
            {items_html}
        </div>
        """, unsafe_allow_html=True)

    with col_gap:
        items_html = ""
        for g in result.gaps:
            items_html += f"""
            <div class="list-item">
                <span class="dot" style="background:#ff9800;"></span>
                <div>
                    <div class="item-point">{g.point}</div>
                    <div class="item-detail">{g.detail}</div>
                </div>
            </div>"""
        st.markdown(f"""
        <div class="result-section">
            <div class="section-title" style="color:#7a4500;">⚡ Area Perlu Ditingkatkan ({len(result.gaps)})</div>
            {items_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <hr>
    <p style='text-align:center;color:#bbb;font-size:0.75rem;'>
        Dianalisis oleh LangChain · Gemini AI · Hasil bersifat rekomendasi
    </p>
    """, unsafe_allow_html=True)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    render_header()

    uploaded, jd_text = render_inputs()
    clicked = render_analyze_button()

    if clicked:
        # Validasi input
        if not uploaded:
            st.error("⚠️ Silakan upload file PDF resume terlebih dahulu.")
            return
        if not jd_text.strip():
            st.error("⚠️ Silakan masukkan Job Description terlebih dahulu.")
            return

        # Step 1: Baca PDF
        with st.spinner("📖 Membaca PDF resume..."):
            file_bytes = uploaded.read()
            resume_text = load_pdf_text(file_bytes)

        if not resume_text:
            st.error("❌ Tidak dapat mengekstrak teks dari PDF. Pastikan PDF tidak terproteksi atau berbasis scan gambar.")
            return

        # Step 2: Jalankan LangChain agent
        with st.spinner("🤖 LangChain agent sedang menganalisis..."):
            try:
                agent = get_agent()
                result = agent.review(resume_text, jd_text)
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat analisis: {str(e)}")
                return

        # Step 3: Tampilkan hasil
        render_results(result)


if __name__ == "__main__":
    main()
