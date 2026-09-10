import streamlit as st
from engine import extract_text_from_pdf, evaluate_candidate

st.set_page_config(page_title="AI Job Match Screener", layout="wide", page_icon="🎯")

st.title("🎯 AI Job Match Screener (RAG + Gemini Agent)")
st.caption("Evaluador técnico de compatibilidad con citas de evidencia y preguntas de entrevista.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Entrada de Datos")
    uploaded_file = st.file_uploader("Cargar CV (Formato PDF)", type=["pdf"])
    job_description = st.text_area("Descripción de la Vacante (Job Description)", height=280, placeholder="Pega aquí los requisitos...")
    
    run_btn = st.button("Evaluar Compatibilidad", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Resultados de la Evaluación")
    
    if run_btn:
        if not uploaded_file or not job_description.strip():
            st.warning("⚠️ Por favor sube un archivo PDF y escribe la descripción de la vacante.")
        else:
            with st.spinner("Procesando embeddings en ChromaDB y evaluando con Gemini..."):
                try:
                    cv_text = extract_text_from_pdf(uploaded_file)
                    result = evaluate_candidate(cv_text, job_description)
                    
                    # Métrica Principal
                    st.metric(label="Score de Compatibilidad", value=f"{result.match_score} %")
                    st.progress(result.match_score / 100)
                    
                    # Resumen
                    st.info(f"**Resumen Ejecutivo:** {result.summary}")
                    
                    # Pestañas
                    tab1, tab2, tab3 = st.tabs(["📋 Evidencia por Requisito", "❓ Preguntas de Entrevista", "💡 Fortalezas / Gaps"])
                    
                    with tab1:
                        for req in result.requirements_analysis:
                            status_color = "🟢" if req.status == "CUMPLE" else ("🟡" if req.status == "PARCIAL" else "🔴")
                            st.markdown(f"**{status_color} {req.requirement}** ({req.status})")
                            st.caption(f"*Cita textual:* \"{req.evidence_quote}\"")
                    
                    with tab2:
                        st.write("Preguntas sugeridas para screening basadas en debilidades:")
                        for q in result.gap_interview_questions:
                            st.write(f"- 🎙️ {q}")
                            
                    with tab3:
                        st.write("**Fortalezas:**")
                        for s in result.key_strengths:
                            st.write(f"✓ {s}")
                        st.write("**Brechas a mitigar:**")
                        for g in result.key_gaps:
                            st.write(f"✗ {g}")
                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")

