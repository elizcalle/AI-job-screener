import streamlit as st
from engine import extract_text_from_pdf, evaluate_candidate

st.set_page_config(
    page_title="Match Jobs | Screener Técnico",
    layout="wide",
    page_icon="🎯"
)

# ----------------- PANEL LATERAL (INPUTS) -----------------
with st.sidebar:
    st.subheader("🎯 Entrada de Datos")
    st.caption("Carga el perfil y los requisitos a contrastar.")
    
    uploaded_file = st.file_uploader("1. CV del Candidato (PDF)", type=["pdf"])
    job_description = st.text_area(
        "2. Descripción del Puesto (JD)",
        height=280,
        placeholder="Pega los requerimientos técnicos y responsabilidades..."
    )
    
    run_btn = st.button("Evaluar Compatibilidad", type="primary", use_container_width=True)

# ----------------- DASHBOARD PRINCIPAL -----------------
st.title("🎯 Dictamen de Compatibilidad Técnica")
st.caption("Auditoría de alineación entre CV y Requerimientos de la Vacante")

if not run_btn:
    st.info("👈 Sube el CV en PDF y la descripción de la vacante en el menú lateral para ejecutar el análisis.")
else:
    if not uploaded_file or not job_description.strip():
        st.warning("⚠️ Debes cargar el archivo PDF y pegar la descripción del puesto.")
    else:
        with st.spinner("Procesando embeddings y contrastando perfil..."):
            try:
                cv_text = extract_text_from_pdf(uploaded_file)
                result = evaluate_candidate(cv_text, job_description)

                # Contadores para el desglose rápido
                c_cumple = sum(1 for r in result.requirements_analysis if r.status == "CUMPLE")
                c_parcial = sum(1 for r in result.requirements_analysis if r.status == "PARCIAL")
                c_nocumple = sum(1 for r in result.requirements_analysis if r.status not in ["CUMPLE", "PARCIAL"])

                # 1. TARJETA KPI SUPERIOR (Nativa con borde)
                with st.container(border=True):
                    col_kpi1, col_kpi2, col_summary = st.columns([1, 1, 2.5])
                    
                    with col_kpi1:
                        st.metric("Índice de Match", f"{result.match_score} %")
                        st.progress(result.match_score / 100)

                    with col_kpi2:
                        st.markdown("**Semáforo General**")
                        st.markdown(f"🟢 **Cumple:** `{c_cumple}`")
                        st.markdown(f"🟡 **Parcial:** `{c_parcial}`")
                        st.markdown(f"🔴 **No Cumple:** `{c_nocumple}`")

                    with col_summary:
                        st.markdown("**Dictamen Ejecutivo**")
                        st.write(result.summary)

                # 2. AUDITORÍA TÉCNICA REQUISITO POR REQUISITO
                st.subheader("📋 Matriz de Verificación de Requisitos")
                
                with st.container(border=True):
                    for req in result.requirements_analysis:
                        # Semáforo limpio con íconos estándar
                        if req.status == "CUMPLE":
                            badge = "🟢 **CUMPLE**"
                        elif req.status == "PARCIAL":
                            badge = "🟡 **PARCIAL**"
                        else:
                            badge = "🔴 **NO CUMPLE**"

                        c_req, c_status = st.columns([4, 1])
                        with c_req:
                            st.markdown(f"**{req.requirement}**")
                            st.caption(f'Evidencia: *"{req.evidence_quote}"*')
                        with c_status:
                            st.markdown(f"<div style='text-align: right;'>{badge}</div>", unsafe_allow_html=True)
                        st.divider()

                # 3. FORTALEZAS VS BRECHAS (Lado a lado en cajas nativas)
                st.subheader("⚖️ Balance de Competencias")
                col_str, col_gap = st.columns(2)

                with col_str:
                    with st.container(border=True):
                        st.markdown("##### 🟢 Fortalezas Acreditadas")
                        for s in result.key_strengths:
                            st.markdown(f"- {s}")

                with col_gap:
                    with st.container(border=True):
                        st.markdown("##### 🔴 Gaps / Brechas a Mitigar")
                        for g in result.key_gaps:
                            st.markdown(f"- {g}")

                # 4. PREPARACIÓN Y AUTOESTUDIO
                st.subheader("🎯 Preparación Técnica para la Entrevista")
                col_prep, col_courses = st.columns([1.2, 1])

                with col_prep:
                    with st.container(border=True):
                        st.markdown("##### 🎙️ Preguntas Clave de Entrevista")
                        st.caption("Enfocadas en tus brechas y arquitectura esperada:")
                        for q in result.gap_interview_questions:
                            st.markdown(f"- **{q}**")
                        st.divider()
                        st.caption("Preguntas de entrevista probables para estudiar:")
                        for q in result.interview_complete:
                            st.markdown(f"- {q}")

                with col_courses:
                    with st.container(border=True):
                        st.markdown("##### 📚 Guía Prioritaria de Estudio")
                        st.caption("Conceptos concretos a repasar antes de la llamada:")
                        for c in result.gap_courses:
                            st.markdown(f"- 📖 {c}")

            except Exception as e:
                st.error(f"Error durante el análisis técnico: {e}")