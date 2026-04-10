"""
Resume Review Agent — dibangun dengan LangChain
Alur: PDF Loader → Text Splitter → LLM Chain → Output Parser
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import List

import pdfplumber
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class StrengthItem:
    point: str
    detail: str


@dataclass
class GapItem:
    point: str
    detail: str


@dataclass
class ReviewResult:
    match_percentage: int
    verdict: str
    summary: str
    strengths: List[StrengthItem]
    gaps: List[GapItem]
    key_skills_matched: List[str]
    key_skills_missing: List[str]
    recommendation: str
    raw_resume_preview: str = ""

    @classmethod
    def from_dict(cls, data: dict, resume_preview: str = "") -> "ReviewResult":
        return cls(
            match_percentage=int(data.get("match_percentage", 0)),
            verdict=data.get("verdict", "-"),
            summary=data.get("summary", ""),
            strengths=[
                StrengthItem(point=s.get("point", ""), detail=s.get("detail", ""))
                for s in data.get("strengths", [])
            ],
            gaps=[
                GapItem(point=g.get("point", ""), detail=g.get("detail", ""))
                for g in data.get("gaps", [])
            ],
            key_skills_matched=data.get("key_skills_matched", []),
            key_skills_missing=data.get("key_skills_missing", []),
            recommendation=data.get("recommendation", ""),
            raw_resume_preview=resume_preview,
        )


# ─── PDF Loader ────────────────────────────────────────────────────────────────

def load_pdf_text(file_bytes: bytes) -> str:
    """Ekstrak teks dari bytes PDF menggunakan pdfplumber."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts).strip()


# ─── Prompt Template ───────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """Anda adalah seorang HR Expert dan Career Coach profesional berpengalaman \
dalam mengevaluasi kesesuaian resume kandidat dengan deskripsi pekerjaan.

Analisis harus objektif, mendalam, dan actionable.

OUTPUT WAJIB berupa JSON valid dengan struktur tepat seperti ini — tanpa teks lain di luar JSON:
{{
    "match_percentage": <integer 0-100>,
    "verdict": "<Sangat Cocok | Cocok | Cukup Cocok | Kurang Cocok | Tidak Cocok>",
    "summary": "<2-3 kalimat ringkasan kandidat dan kecocokannya>",
    "strengths": [
        {{"point": "<judul kekuatan>", "detail": "<penjelasan 1 kalimat>"}},
        ...
    ],
    "gaps": [
        {{"point": "<judul kekurangan>", "detail": "<penjelasan 1 kalimat>"}},
        ...
    ],
    "key_skills_matched": ["<skill>", ...],
    "key_skills_missing": ["<skill>", ...],
    "recommendation": "<saran konkret dalam 2-3 kalimat>"
}}

Aturan:
- match_percentage harus realistis berdasarkan analisis faktual
- strengths: 3 hingga 6 poin
- gaps: 2 hingga 5 poin
- Semua nilai string dalam Bahasa Indonesia
- HANYA JSON, tidak ada markdown, tidak ada preamble"""

HUMAN_TEMPLATE = """## RESUME KANDIDAT
{resume_text}

## JOB DESCRIPTION
{job_description}

Analisis kesesuaian dan kembalikan JSON."""


# ─── Output Parser ─────────────────────────────────────────────────────────────

def parse_json_output(raw: str) -> dict:
    """Bersihkan dan parse output JSON dari LLM."""
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    return json.loads(cleaned)


# ─── LangChain Agent ───────────────────────────────────────────────────────────

class ResumeReviewAgent:
    """
    Agent review resume berbasis LangChain LCEL (LangChain Expression Language).

    Pipeline:
        input dict
          → ChatPromptTemplate   (format prompt)
          → ChatGoogleGenerativeAI (panggil LLM)
          → StrOutputParser      (ambil teks)
          → parse_json_output    (parse ke dict)
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        api_key: str | None = None,
    ):
        api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key tidak ditemukan. Setel environment variable `GOOGLE_API_KEY` atau `GEMINI_API_KEY`, "
                "atau panggil ResumeReviewAgent(api_key='...')."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            ("human", HUMAN_TEMPLATE),
        ])

        # LCEL chain: prompt | llm | str parser | json parser
        self.chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
            | RunnableLambda(parse_json_output)
        )

    def review(self, resume_text: str, job_description: str) -> ReviewResult:
        """
        Jalankan review resume.

        Args:
            resume_text: Teks hasil ekstraksi PDF
            job_description: Job description dari pengguna

        Returns:
            ReviewResult dataclass berisi hasil analisis lengkap
        """
        result_dict = self.chain.invoke({
            "resume_text": resume_text,
            "job_description": job_description,
        })

        preview = resume_text[:300] + "..." if len(resume_text) > 300 else resume_text
        return ReviewResult.from_dict(result_dict, resume_preview=preview)
