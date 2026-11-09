# 🎯 AI Job Match Screener (RAG & Gemini Pipeline)

An executive-grade technical screening engine that audits candidate resumes (PDFs) against job descriptions using Retrieval-Augmented Generation (RAG), local semantic search, and structured schema outputs powered by Google Gemini.

> **Live Demo:** [Deploy on Streamlit Community Cloud](https://streamlit.io/)

---

## 🚀 Key Features

* **PDF Ingestion & Chunking:** Ingests candidate resumes using `pypdf`, normalizing text into logical semantic chunks.
* **In-Memory RAG Architecture:** Employs `ChromaDB` for instant local vector indexing and top-k context retrieval against job description requirements without external database overhead.
* **Audited Grounding & Zero-Hallucination:** Forces the LLM to extract verbatim quotes (`evidence_quote`) from the candidate's CV to support each requirement's compliance status.
* **Requirements Compliance Matrix:** Instant traffic-light classification (`CUMPLE` / `PARCIAL` / `NO CUMPLE`) across technical qualifications and core competencies.
* **Strict Typed Outputs:** Leverages `Pydantic v2` schemas to guarantee structured JSON output parsing (`match_score`, `key_strengths`, `key_gaps`, targeted interview questions, and study guides).
* **Gap Mitigation & Interview Prep:** Generates role-specific behavioral/technical questions focused on identified deficiencies, plus a curated self-study roadmap to bridge technical gaps.
* **Resilient API Calls:** Integrated exponential backoff and jitter via `tenacity` to handle rate limits (`429`) and upstream service saturation gracefully.
* **Modern Executive Dashboard:** Built with native `Streamlit` containers for clean responsive layouts in both light and dark themes.

---

## 🛠️ Tech Stack & Architecture

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend / UI** | `Streamlit` | Interactive data input, real-time metrics, and audit dashboards |
| **LLM Inference** | `Google Gemini` (`gemini-2.5-flash`) | Context analysis, structured extraction, and interview synthesis |
| **Vector Engine** | `ChromaDB` | In-memory embedding generation and semantic similarity search |
| **Data Validation** | `Pydantic v2` | Strict schema enforcement (`RequirementMatch`, `ScreeningResult`) |
| **Document Parser** | `PyPDF` | Multi-page text extraction from uploaded PDF resumes |
| **Resilience** | `Tenacity` | Exponential backoff retry handler for API rate limits |

---

## ⚙️ Local Setup & Installation

### 1. Clone the repository
```bash
git clone [https://github.com/elizcalle/AI-job-screener.git](https://github.com/elizcalle/AI-job-screener.git)
cd AI-job-screener