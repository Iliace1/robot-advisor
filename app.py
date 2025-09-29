import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import openai
import os

# --- CONFIG OPENAI ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- TITRE ---
st.title("📊 Robot Advisor – Optimisation & Conseil Intelligent")

# --- UPLOAD CSV ---
uploaded_file = st.file_uploader("Importez vos données ETF (CSV)", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.info("⚠️ Aucun fichier uploadé. Chargement du fichier par défaut.")
    data = pd.read_csv("etf_fusionne_vertical.csv")

# --- APERCU ---
st.subheader("Aperçu :")
st.dataframe(data.head())

# --- CHOIX ETF ---
colonnes = list(data.columns[1:])
choix_etf = st.multiselect("📌 Choisissez vos ETF :", colonnes, default=colonnes[:3])

# --- CHOIX BENCHMARK ---
benchmark = st.selectbox("📊 Choisissez un ETF benchmark :", colonnes)

# --- PROFIL ---
profil = st.radio("👤 Choisissez votre profil :", ["Défensif", "Équilibré", "Offensif"])

# --- ANALYSE ---
if choix_etf:
    rendements = data[choix_etf].pct_change().dropna()
    stats = rendements.describe().T
    stats["Moyenne"] = rendements.mean()
    stats["Volatilité"] = rendements.std()
    stats["Sharpe"] = stats["Moyenne"] / stats["Volatilité"]

    st.subheader("📈 Statistiques :")
    st.write(stats[["Moyenne", "Volatilité", "Sharpe"]])

    # --- GRAPHIQUE ---
    st.subheader("📊 Evolution des ETF sélectionnés")
    plt.figure(figsize=(10, 5))
    for etf in choix_etf:
        plt.plot(data["Date"], data[etf], label=etf)
    plt.legend()
    st.pyplot(plt)

    # --- RAPPORT TEXTE ---
    st.subheader("📝 Rapport")
    rapport = f"Profil choisi : {profil}\n\nETF sélectionnés : {', '.join(choix_etf)}"
    st.text_area("Rapport généré :", rapport, height=150)

    # --- PDF ---
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=14)
        pdf.cell(200, 10, "Rapport Robot Advisor", ln=True, align="C")
        pdf.set_font("Helvetica", size=12)
        pdf.cell(200, 10, f"Profil : {profil}", ln=True)
        pdf.ln(10)
        safe_text = rapport.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 10, safe_text)

        # --- FIX : conversion en bytes ---
        pdf_output = pdf.output(dest="S")
        if isinstance(pdf_output, str):
            return pdf_output.encode("latin-1", "ignore")
        elif isinstance(pdf_output, bytearray):
            return bytes(pdf_output)
        return pdf_output

    pdf_bytes = generate_pdf()
    st.download_button(
        "📥 Télécharger le rapport PDF",
        data=pdf_bytes,
        file_name="rapport_robot_advisor.pdf",
        mime="application/pdf"
    )

    # --- CONSEILLER IA ---
    st.subheader("🤖 Conseiller IA")
    user_question = st.text_input("Posez une question à l’IA sur vos ETF ou votre portefeuille :")
    if user_question:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tu es un conseiller financier spécialisé en ETF."},
                    {"role": "user", "content": user_question}
                ]
            )
            st.success(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Erreur IA : {e}")
