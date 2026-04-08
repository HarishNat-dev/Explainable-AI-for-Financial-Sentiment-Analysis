from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

import streamlit as st

# --- Make project root importable when Streamlit runs from src/app ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import FinBERTPredictor, InferenceConfig
from src.xai.ig_explainer import FinBERTIGExplainer, IGConfig
from src.xai.shap_explainer import FinBERTSHAPExplainer, SHAPConfig
from src.app.batch_utils import guess_text_column, run_batch_predictions, add_top_ig_tokens



st.set_page_config(
    page_title="Financial Sentiment + Explainability (FinBERT)",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Financial Sentiment Analysis (FinBERT) + Explainability")
st.caption("Fine-tuned FinBERT on Financial PhraseBank (AllAgree). XAI: Integrated Gradients + SHAP.")

# Cache model + explainers so they load once
@st.cache_resource
def load_predictor():
    return FinBERTPredictor(InferenceConfig())

@st.cache_resource
def load_ig_explainer():
    return FinBERTIGExplainer(IGConfig(n_steps=30))

@st.cache_resource
def load_shap_explainer():
    # If SHAP is slow on CPU, reduce max_evals (150–250 is fine)
    return FinBERTSHAPExplainer(SHAPConfig(max_evals=250))

predictor = load_predictor()
ig_explainer = load_ig_explainer()
shap_explainer = load_shap_explainer()

default_text = "Company shares rise after strong earnings report."
text = st.text_area("Enter a financial headline / sentence", value=default_text, height=90)

tab_pred, tab_ig, tab_shap, tab_batch = st.tabs(["Prediction", "Integrated Gradients", "SHAP", "Batch CSV"])

with tab_pred:
    st.subheader("Prediction")
    if st.button("Analyze", type="primary"):
        pred = predictor.predict_one(text)
        st.metric("Sentiment", pred["label"].upper())
        st.metric("Confidence", f"{pred['confidence']:.3f}")
        st.write("Class probabilities:")
        st.json(pred["probabilities"])

with tab_ig:
    st.subheader("Integrated Gradients")
    st.write("Green tokens support the explained class; red tokens oppose it.")

    if st.button("Explain with IG"):
        out = ig_explainer.explain(text)

        st.markdown(
            f"**Explaining:** `{out['target_explained']['label'].upper()}` "
            f"(predicted: `{out['predicted']['label'].upper()}` | confidence: `{out['predicted']['confidence']:.3f}`)"
        )

        st.markdown(out["html"], unsafe_allow_html=True)

        pairs = [(t, a) for t, a in zip(out["tokens"], out["attributions"]) if str(t).strip() != ""]
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:12]
        st.write("Top tokens by |attribution|:")
        st.table([{"token": t, "attribution": float(a)} for t, a in pairs_sorted])

with tab_shap:
    st.subheader("SHAP")
    st.write("Green tokens support the explained class; red tokens oppose it.")
    st.caption("Tip: If SHAP feels slow on CPU, reduce max_evals in SHAPConfig (e.g., 150–250).")

    if st.button("Explain with SHAP"):
        out = shap_explainer.explain(text)

        st.markdown(
            f"**Explaining:** `{out['target_explained']['label'].upper()}` "
            f"(predicted: `{out['predicted']['label'].upper()}` | confidence: `{out['predicted']['confidence']:.3f}`)"
        )

        st.markdown(out["html"], unsafe_allow_html=True)

        pairs = [(t, a) for t, a in zip(out["tokens"], out["attributions"]) if str(t).strip() != ""]
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:12]
        st.write("Top tokens by |SHAP|:")
        st.table([{"token": t, "shap_value": float(a)} for t, a in pairs_sorted])


with tab_batch:
    st.subheader("Batch CSV Upload")
    st.write("Upload a CSV of headlines/sentences and get predictions + confidence, plus Top-IG tokens for explanations.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

        st.write("Preview:")
        st.dataframe(df.head(10), use_container_width=True)

        guessed = guess_text_column(df)
        text_col = st.selectbox(
            "Select the column that contains the text/headlines",
            options=list(df.columns),
            index=(list(df.columns).index(guessed) if guessed in df.columns else 0),
        )

        max_rows = st.number_input(
            "Max rows to run (safety)",
            min_value=1,
            max_value=len(df),
            value=min(len(df), 500),
        )

        df_run = df.head(int(max_rows)).copy()

        # --- IG explanation controls ---
        explain_ig = st.checkbox("Add Top IG tokens to output (slower)", value=True)
        top_k = st.slider("Top-K IG tokens", min_value=3, max_value=12, value=5)
        max_explain = st.number_input(
            "Max rows to explain with IG (speed control)",
            min_value=1,
            max_value=int(max_rows),
            value=min(int(max_rows), 100),
        )

        if st.button("Run batch predictions", type="primary"):
            results = run_batch_predictions(df_run, text_col=text_col, predictor=predictor)

            if explain_ig:
                with st.spinner("Computing Integrated Gradients explanations (this may take a bit on CPU)..."):
                    results = add_top_ig_tokens(
                        results,
                        text_col=text_col,
                        ig_explainer=ig_explainer,
                        top_k=int(top_k),
                        max_rows_to_explain=int(max_explain),
                    )

            st.success(f"Done! Scored {len(results)} rows.")

            # Summary
            c1, c2 = st.columns(2)
            with c1:
                st.write("Sentiment distribution:")
                st.bar_chart(results["sentiment"].value_counts())
            with c2:
                st.write("Confidence summary:")
                st.write(results["confidence"].describe())

            st.write("Results:")
            st.dataframe(results, use_container_width=True)

            # Download
            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                data=csv_bytes,
                file_name="batch_sentiment_results.csv",
                mime="text/csv",
            )
