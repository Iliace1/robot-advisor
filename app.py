import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
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
    st.warning("⚠️ Pas de données disponibles pour la partie portefeuille. "
               "Vous pouvez quand même utiliser le Conseiller IA plus bas.")
    raw = None  # IA restera utilisable

# ------------------------------
# Analyse portefeuille (si data)
# ------------------------------
if raw is not None:
    # Pivot pour obtenir les ETF en colonnes, valeurs = Close
    data = raw.pivot(index="Date", columns="ETF", values="Close")
    data = data.dropna(axis=1, how="all")  # supprime colonnes vides
    st.write("Aperçu :", data.head())

    # Rendements (jour -> jour), lignes NaN supprimées
    returns = data.pct_change().dropna()

    # ------------------------------
    # Sélection ETF + Benchmark
    # ------------------------------
    etfs = list(returns.columns)
    choix_etfs = st.multiselect("📌 Choisissez vos ETF :", etfs, default=etfs[:3])
    benchmark_etf = st.selectbox("📈 Choisissez un ETF benchmark :", etfs)

    if not choix_etfs:
        st.warning("Sélectionnez au moins un ETF pour la partie portefeuille.")
    else:
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

        cov_annual = sel_returns.cov() * 252
        mean_daily = sel_returns.mean()

        for i in range(n_portfolios):
            w = np.random.random(len(choix_etfs))
            w /= np.sum(w)
            weights_record.append(w)

            port_ret = np.dot(w, mean_daily) * 252
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_annual, w)))
            sharpe = port_vol and (port_ret / port_vol) or np.nan

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
        else:
            max_sharpe_port = portfolios.loc[portfolios["Sharpe"].idxmax()]
            min_risk_port = portfolios.loc[portfolios["Risque"].idxmin()]

            # ------------------------------
            # Graphe Frontière efficiente
            # ------------------------------
            st.subheader("🎯 Optimisation de portefeuille")
            fig = px.scatter(portfolios, x="Risque", y="Rendement", color="Sharpe",
                             hover_data=choix_etfs, title="Frontière efficiente (Monte Carlo)")
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
            # Export PDF (robuste)
            # ------------------------------
            def generate_pdf_bytes(txt: str) -> bytes:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", size=14)
                pdf.cell(200, 10, "Rapport Robot Advisor", ln=True, align="C")
                pdf.set_font("Helvetica", size=12)
                pdf.ln(2)
                pdf.cell(200, 10, f"Profil : {profil}", ln=True)
                pdf.ln(6)
                # Texte (encodage sûr)
                safe_text = txt
                try:
                    safe_text.encode("latin-1")
                except Exception:
                    safe_text = txt.encode("latin-1", "replace").decode("latin-1")
                pdf.multi_cell(0, 8, safe_text)

                # Normaliser en bytes (str/bytes/bytearray compat)
                out = pdf.output(dest="S")
                if isinstance(out, bytes):
                    return out
                if isinstance(out, bytearray):
                    return bytes(out)
                # str -> bytes latin-1
                return out.encode("latin-1", "replace")

            pdf_bytes = generate_pdf_bytes(rapport)
            st.download_button(
                "📥 Télécharger le rapport PDF",
                data=pdf_bytes,
                file_name="rapport_robot_advisor.pdf",
                mime="application/pdf",
            )

# ---------------------------------
# 💬 Conseiller IA — toujours visible
# ---------------------------------
st.subheader("💬 Conseiller IA (chat)")

api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

with st.expander("ℹ️ Comment activer l'IA ?", expanded=False):
    st.markdown(
        """
        - Va dans **Manage app ▸ Secrets** et ajoute :
          ```
          OPENAI_API_KEY = "ta_cle_api_openai"
          ```
        - Puis **Rerun** l’application.
        """
    )

prompt = st.text_area("Posez une question à l'IA :", "Propose une stratégie adaptée à mon profil.")
go = st.button("Obtenir la réponse IA")

if go:
    if not api_key:
        st.warning("⚠️ Pas de clé API configurée dans *Secrets* (`OPENAI_API_KEY`).")
    else:
        try:
            client = OpenAI(api_key=api_key)

            # Contexte minimal si pas de data
            ctx = {}
            try:
                ctx = {
                    "colonnes": list(raw.columns) if raw is not None else None,
                }
            except Exception:
                pass

            with st.spinner("L’IA réfléchit..."):
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system",
                         "content": "Tu es un conseiller financier professionnel, prudent et pédagogique. "
                                    "Donne des explications claires et actionnables."},
                        {"role": "user",
                         "content": f"Contexte: {ctx}. Question: {prompt}"}
                    ],
                    max_tokens=500,
                    temperature=0.4,
                )
            st.success("✅ Réponse IA :")
            st.markdown(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Erreur API : {e}")
