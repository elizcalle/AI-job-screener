# 🎯 AI Job Match Screener (RAG & LLM Pipeline)

An intelligent technical screening engine that analyzes candidate resumes (PDFs) against job descriptions using **Retrieval-Augmented Generation (RAG)**, local semantic vector search, and structured outputs powered by **Gemini 3.5**.

---

## 🚀 Key Features

* **PDF Ingestion & Text Extraction**: Efficiently extracts raw resume text using `pypdf`.
* **RAG Pipeline with ChromaDB**: Segments resumes into logical chunks and indexes them in an in-memory vector database for semantic similarity retrieval.
* **Grounding & Hallucination Mitigation**: Constrains the LLM to extract verbatim textual evidence (`evidence_quote`) from the candidate's CV.
* **Strict Structured Outputs**: Utilizes **Pydantic** to enforce rigorous data validation and return typed JSON schemas (`match_score`, `key_strengths`, `key_gaps`, and technical interview questions).
* **Interactive UI**: Clean, reactive web interface built with **Streamlit**.

---

## 🛠️ Tech Stack & Architecture

* **Language**: Python 3.10+
* **LLM**: Google Gemini (`gemini-3.5-flash`) via `google-genai` SDK
* **Vector Database**: ChromaDB (In-memory semantic search)
* **Schema Validation**: Pydantic v2
* **Frontend**: Streamlit
* **Document Processing**: PyPDF

---

## ⚙️ Local Setup & Installation

### 1. Clone the repository
```bash
git clone [https://github.com/elizcalle/AI-job-screener.git](https://github.com/elizcalle/AI-job-screener.git)
cd AI-job-screener