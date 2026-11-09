import os
import time
from typing import List
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field
from pypdf import PdfReader
import streamlit as st
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Cargar variables locales (.env) si existen
load_dotenv()

def get_gemini_client():
    # 1. Busca en variables de entorno (local)
    key = os.getenv("GEMINI_API_KEY")
    
    # 2. Si no está en entorno, busca en secrets de Streamlit
    if not key:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    if not key:
        raise ValueError("No se encontró GEMINI_API_KEY en las variables ni en Secrets.")

    return genai.Client(api_key=key)

chroma_client = chromadb.Client()


# 1. Esquema de datos estructurado (Pydantic)
class RequirementMatch(BaseModel):
    requirement: str = Field(description="El requisito específico de la vacante")
    status: str = Field(description="'CUMPLE', 'PARCIAL' o 'NO CUMPLE'")
    evidence_quote: str = Field(
        description="Cita textual exacta del CV que respalda la evaluación"
    )


class ScreeningResult(BaseModel):
    match_score: int = Field(
        description="Porcentaje de compatibilidad general entre 0 y 100"
    )
    summary: str = Field(
        description="Resumen ejecutivo del perfil frente a la vacante"
    )
    requirements_analysis: List[RequirementMatch] = Field(
        description="Desglose con evidencia textual"
    )
    key_strengths: List[str] = Field(
        description="Fortalezas principales identificadas"
    )
    key_gaps: List[str] = Field(
        description="Brechas técnicas o de experiencia"
    )
    gap_interview_questions: List[str] = Field(
        description="3 a 5 preguntas técnicas basadas en las brechas para evaluar al candidato"
    )
    interview_complete: List[str] = Field(
        description="10 a 15 preguntas completas basadas en la descripcion de la vacante para evaluar al candidato."
    )
    gap_courses: List[str] = Field(
        description="3 a 5 contenido en internet para fortalecer las debilidades del candidato"
    )


# 2. Extracción de texto desde archivo PDF
def extract_text_from_pdf(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


# 3. Pipeline RAG: Indexar y buscar directamente con ChromaDB
def query_rag_cv(cv_text: str, job_description: str) -> str:
    try:
        chroma_client.delete_collection("cv_collection")
    except Exception:
        pass

    collection = chroma_client.create_collection("cv_collection")

    chunks = [c.strip() for c in cv_text.split("\n\n") if len(c.strip()) > 30]
    if not chunks:
        chunks = [c.strip() for c in cv_text.split("\n") if len(c.strip()) > 30]

    if not chunks:
        return cv_text[:2000]

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks
    )

    results = collection.query(
        query_texts=[job_description[:1000]],
        n_results=min(5, len(chunks))
    )

    retrieved_context = "\n---\n".join(results["documents"][0])
    return retrieved_context


# 4. Llamada con Reintento Exponencial
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(APIError),
    reraise=True,
)
def call_gemini_with_retry(client, model_name, prompt, schema):
    return client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
        ),
    )


# 5. Agente Evaluador Gemini con Structured Outputs
def evaluate_candidate(cv_text: str, job_description: str) -> ScreeningResult:
    client = get_gemini_client()  # <- Se inicializa aquí de forma segura
    context_chunks = query_rag_cv(cv_text, job_description)

    prompt = f"""Eres un Reclutador Técnico Senior y Especialista en Selección de Talento...
    
VACANTE:
{job_description}

EVIDENCIA RECUPERADA DEL CV (Vía Búsqueda Vectorial RAG):
{context_chunks}

TEXTO COMPLETO DEL CV (Para contexto general):
{cv_text}

REGLAS ESTRICTAS DE EVALUACIÓN:
1. Citas Textuales Obligatorias: Para cada requisito analizado, incluye la frase textual exacta del CV en 'evidence_quote'. Si no cumple, coloca 'No se encontró mención en el documento'.
2. Preguntas de Entrevista por Brechas: Genera entre 3 y 5 preguntas técnicas profundas dirigidas específicamente a validar si el candidato puede compensar las brechas (gaps) encontradas.
3. Rigor: Asigna el match_score de 0 a 100 de acuerdo al cumplimiento de Must-haves vs Nice-to-haves.
4. Entrevista: Genera una guía de 10 a 15 preguntas de entrevista de acuerdo a los requerimientos de la vacante para estudiar y asegurar mejor el puesto.
5. Estudio: Recomienda cursos o guías de estudio concretas para nivelar lo que NO CUMPLE el candidato frente a la vacante."""

    response = call_gemini_with_retry(
        client=client,
        model_name="gemini-2.5-flash",
        prompt=prompt,
        schema=ScreeningResult,
    )

    return ScreeningResult.model_validate_json(response.text)