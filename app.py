import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
import os
from openai import OpenAI

# ------------------------------
# Config
# ------------------------------
st.set_page_config(page_title="📊 Financial Advisor", layout="wide")
st.title("📊 Robot Advisor – Optimisation & Conseil Intelligent")

# ------------------------------
# Import Data (avec fallback)
# ------------------------------
DEFAULT_FILE = "etf_fusionne_vertical.csv"

uploaded_file = st.file_uploader("📂 Importez vos données ETF (CSV)", type=["csv"])

if uploaded_file:
    raw = pd.read_csv(uploaded_file, parse_dates=["Date"])
    st.success("✅ Données chargées depuis le fichier uploadé")
elif os.path.exists(DEFAULT_FILE):
    raw = pd.read_csv(DEFAULT_FILE, parse_dates=["Date"])
    st.info("ℹ️ Aucun fichier uploadé. Chargement du fichier par défaut.")
else:
    st.error("❌ Pas de données disponibles. Importez un fichier CSV.")
    st.stop()

# Pivot pour obtenir les ETF en colonnes, valeurs = Close
data = raw.pivot(index="Date", columns="ETF", values="Close")
data = data.dropna(axis=1, how="all")  # supprime colonnes vides
st.write("Aperçu :", data.head())

returns = data.pct_change().dropna()

# ------------------------------
# Sélection ETF + Benchmark
# ------------------------------
etfs = list(returns.columns)
choix_etfs = st.multiselect("📌 Choisissez vos ETF :", etfs, default=etfs[:3])
benchmark_etf = st.selectbox("📈 Choisissez un ETF benchmark :", etfs)

if not choix_etfs:
    st.warning("Sélectionnez au moins un ETF.")
    st.stop()

sel_returns = returns[choix_etfs]

# ------------------------------
# Profil Investisseur
# ------------------------------
profil = st.radio("👤 Choisissez votre profil :", ["Défensif", "Équilibré", "Offensif"])

# ------------------------------
# Stats descriptives
# ------------------------------
st.subheader("📈 Statistiques descriptives")
stats = pd.DataFrame({
    "Rendement moyen": sel_returns.mean() * 252,
    "Volatilité": sel_returns.std() * np.sqrt(252),
    "Sharpe brut": (sel_returns.mean() * 252) / (sel_returns.std() * np.sqrt(252)),
    "Skew": sel_returns.skew(),
    "Kurtosis": sel_returns.kurt()
})
st.dataframe(stats.style.format("{:.2%}"))

# ------------------------------
# Simulation Monte Carlo
# ------------------------------
n_portfolios = 5000
results = np.zeros((3, n_portfolios))
weights_record = []

for i in range(n_portfolios):
    weights = np.random.random(len(choix_etfs))
    weights /= np.sum(weights)
    weights_record.append(weights)

    port_ret = np.dot(weights, sel_returns.mean()) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(sel_returns.cov() * 252, weights)))
    sharpe = port_ret / port_vol if port_vol > 0 else np.nan

    results[0, i] = port_ret
    results[1, i] = port_vol
    results[2, i] = sharpe

portfolios = pd.DataFrame(weights_record, columns=choix_etfs)
portfolios["Rendement"] = results[0]
portfolios["Risque"] = results[1]
portfolios["Sharpe"] = results[2]
portfolios = portfolios.dropna()

if portfolios.empty:
    st.error("⚠️ Aucun portefeuille valide généré.")
    st.stop()

max_sharpe_port = portfolios.loc[portfolios["Sharpe"].idxmax()]
min_risk_port = portfolios.loc[portfolios["Risque"].idxmin()]

# ------------------------------
# Graphe Frontière efficiente
# ------------------------------
st.subheader("🎯 Optimisation de portefeuille")
fig = px.scatter(portfolios, x="Risque", y="Rendement", color="Sharpe", hover_data=choix_etfs)
fig.add_scatter(x=[max_sharpe_port["Risque"]], y=[max_sharpe_port["Rendement"]],
                mode="markers+text", text=["Max Sharpe"], marker=dict(color="red", size=12))
fig.add_scatter(x=[min_risk_port["Risque"]], y=[min_risk_port["Rendement"]],
                mode="markers+text", text=["Min Risk"], marker=dict(color="blue", size=12))
st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Backtest & Benchmark (séparés)
# ------------------------------
st.subheader("⏳ Backtest & Benchmark")
opt_weights = max_sharpe_port[choix_etfs].values
opt_returns = (sel_returns * opt_weights).sum(axis=1)
cum_opt = (1 + opt_returns).cumprod()
cum_bench = (1 + returns[benchmark_etf]).cumprod()

# Graphe 1 : Portefeuille optimal
fig_backtest = px.line(cum_opt, title="📊 Croissance cumulée du Portefeuille Optimal")
fig_backtest.update_layout(yaxis_title="Croissance cumulée", xaxis_title="Date")
st.plotly_chart(fig_backtest, use_container_width=True)

# Graphe 2 : Benchmark
fig_bench = px.line(cum_bench, title=f"📈 Croissance cumulée du Benchmark ({benchmark_etf})")
fig_bench.update_layout(yaxis_title="Croissance cumulée", xaxis_title="Date")
st.plotly_chart(fig_bench, use_container_width=True)

# ------------------------------
# Rapport Auto Profil
# ------------------------------
st.subheader("📄 Rapport automatique")

rapport = f"### Portefeuille Diversifié pour Investisseur {profil}\n\n"
rapport += "#### 1. Allocation des Actifs\n"
for etf, w in zip(choix_etfs, max_sharpe_port[choix_etfs]):
    rapport += f"- **{etf} : {w:.1%}**  \n"
    rapport += f"  - Rendement moyen : {stats.loc[etf, 'Rendement moyen']:.2%}  \n"
    rapport += f"  - Volatilité : {stats.loc[etf, 'Volatilité']:.2%}  \n"
    rapport += f"  - Sharpe brut : {stats.loc[etf, 'Sharpe brut']:.2f}  \n"

rapport += "\n#### 2. Justification de l’Allocation\n"
if profil == "Défensif":
    rapport += "- Accent sur la stabilité et réduction du risque.\n- Pondération plus élevée en obligations/cash.\n"
elif profil == "Équilibré":
    rapport += "- Compromis rendement/risque.\n- Diversification équilibrée.\n"
else:
    rapport += "- Accent sur la croissance.\n- Plus forte pondération sur ETF volatils.\n"

rapport += "\n#### 3. Suivi & Rééquilibrage\n"
rapport += "- Vérifier trimestriellement.\n- Rééquilibrer si un ETF dérive de ±5%.\n"

st.markdown(rapport)

# ------------------------------
# Export PDF (UTF-8 safe)
# ------------------------------
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
    return pdf.output(dest="S").encode("latin-1", "replace")

pdf_bytes = generate_pdf()
st.download_button("📥 Télécharger le rapport PDF",
                   data=pdf_bytes,
                   file_name="rapport_robot_advisor.pdf",
                   mime="application/pdf")

# ------------------------------
# IA Conseiller
# ------------------------------
st.subheader("💬 Conseiller IA")
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
prompt = st.text_area("Posez une question à l'IA :", "Propose une stratégie adaptée à mon profil.")

if st.button("Obtenir la réponse IA"):
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            with st.spinner("L’IA réfléchit..."):
                context = f"Profil : {profil}, ETF : {choix_etfs}, Stats : {stats.to_dict()}"
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Tu es un conseiller financier professionnel et pédagogue."},
                        {"role": "user", "content": f"Contexte : {context}"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400,
                    temperature=0.4
                )
            st.success("✅ Réponse IA :")
            st.markdown(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Erreur API : {e}")
    else:
        st.warning("⚠️ Pas de clé API configurée.")
