import os
from typing import List
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from pypdf import PdfReader

# Cargar las variables del archivo .env
load_dotenv()

# Inicializar clientes
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.Client()


# 1. Esquema de datos estructurado (Pydantic)
class RequirementMatch(BaseModel):
    requirement: str = Field(
        description="El requisito específico de la vacante"
    )
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


# 2. Extracción de texto desde archivo PDF
def extract_text_from_pdf(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text  # Tabulado afuera del for para leer todas las páginas


# 3. Pipeline RAG: Indexar y buscar directamente con ChromaDB
def query_rag_cv(cv_text: str, job_description: str) -> str:
    # Reiniciar colección temporal en ChromaDB
    try:
        chroma_client.delete_collection("cv_collection")
    except Exception:
        pass

    collection = chroma_client.create_collection("cv_collection")

    # Dividir el CV en bloques lógicos de texto (chunking)
    chunks = [c.strip() for c in cv_text.split("\n\n") if len(c.strip()) > 30]
    if not chunks:
        chunks = [c.strip() for c in cv_text.split("\n") if len(c.strip()) > 30]

    # ChromaDB genera los vectores automáticamente sin llamar a la API de embeddings
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))], documents=chunks
    )

    # Búsqueda semántica directa
    results = collection.query(
        query_texts=[job_description[:1000]], n_results=min(5, len(chunks))
    )

    retrieved_context = "\n---\n".join(results["documents"][0])
    return retrieved_context


# 4. Agente Evaluador Gemini con Structured Outputs
def evaluate_candidate(cv_text: str, job_description: str) -> ScreeningResult:
    context_chunks = query_rag_cv(cv_text, job_description)

    prompt = f"""Eres un Reclutador Técnico Senior y Líder de Ingeniería. Tu tarea es evaluar con rigor la compatibilidad de un candidato frente a una descripción de vacante.

    VACANTE:
    {job_description}

    EVIDENCIA RECUPERADA DEL CV (Vía Búsqueda Vectorial RAG):
    {context_chunks}

    TEXTO COMPLETO DEL CV (Para contexto general):
    {cv_text}

    REGLAS ESTRICTAS DE EVALUACIÓN:
    1. Citas Textuales Obligatorias: Para cada requisito analizado, incluye la frase textual exacta del CV en 'evidence_quote'. Si no cumple, coloca 'No se encontró mención en el documento'.
    2. Preguntas de Entrevista por Brechas: Genera entre 3 y 5 preguntas técnicas profundas dirigidas específicamente a validar si el candidato puede compensar las brechas (gaps) encontradas.
    3. Rigor: Asigna el match_score de 0 a 100 de acuerdo al cumplimiento de Must-haves vs Nice-to-haves."""

    response = client.models.generate_content(
        #model="gemini-3.6-flash",
        model = "gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ScreeningResult,
            temperature=0.2,
        ),
    )

    return ScreeningResult.model_validate_json(response.text)