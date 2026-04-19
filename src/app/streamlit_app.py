# =============================================================================
# src/app/streamlit_app.py
#
# Main Streamlit dashboard — the interactive user-facing component of the
# XAI Financial Sentiment platform (Section 4.7, Objective 5).
#
# Organises the interface into four tabs:
#   1. Prediction       — sentiment label + class probabilities (Section 4.7.1)
#   2. Integrated Gradients — token attribution via IG (Section 4.7.2)
#   3. SHAP             — token attribution via SHAP (Section 4.7.3)
#   4. Batch CSV        — bulk prediction + optional IG explanation (Section 4.7.4)
#
# Key design decision: all three model components (predictor, IG explainer,
# SHAP explainer) are loaded once at startup via @st.cache_resource and
# reused across all interactions. Without this the 440 MB FinBERT weights
# would reload on every button click (Section 4.7.5).
#
# Used by: run with `streamlit run src/app/streamlit_app.py`
# Report:  Section 4.7 — Streamlit Application
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

import streamlit as st


# Ensure project root is on the path so all src.* imports resolve correctly
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


# ---------------------------------------------------------------------------
# Model loading — cached so the 440 MB FinBERT weights load only once per
# session. This is the most consequential single decision in the application
# layer (Section 4.7.5). Removing @st.cache_resource would add 10-30s of
# latency to every button click.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_predictor():
    return FinBERTPredictor(InferenceConfig())

@st.cache_resource
def load_ig_explainer():
    # n_steps=30 chosen as efficiency-accuracy trade-off (Section 4.5.1):
    # increasing to 50-100 steps changes attributions by <0.005 on tested sentences
    return FinBERTIGExplainer(IGConfig(n_steps=30))

@st.cache_resource
def load_shap_explainer():
    # max_evals=250 for real-time use; evaluation scripts use 300 (Section 4.5.2)
    return FinBERTSHAPExplainer(SHAPConfig(max_evals=250))

predictor = load_predictor()
ig_explainer = load_ig_explainer()
shap_explainer = load_shap_explainer()


# Shared text input — persists across all four tabs via st.session_state
default_text = "Company shares rise after strong earnings report."
text = st.text_area("Enter a financial headline / sentence", value=default_text, height=90)

tab_pred, tab_ig, tab_shap, tab_batch = st.tabs(["Prediction", "Integrated Gradients", "SHAP", "Batch CSV"])


# ---------------------------------------------------------------------------
# Tab 1: Prediction (Section 4.7.1)
# Returns label, confidence score, and full three-class probability breakdown.
# Displaying all three probabilities rather than just the winning label is a
# deliberate design decision — POSITIVE at 0.998 carries different analytical
# weight than POSITIVE at 0.62 (Section 4.7.1).
# ---------------------------------------------------------------------------
with tab_pred:
    st.subheader("Prediction")
    if st.button("Analyze", type="primary"):
        pred = predictor.predict_one(text)
        st.metric("Sentiment", pred["label"].upper())
        st.metric("Confidence", f"{pred['confidence']:.3f}")
        st.write("Class probabilities:")
        st.json(pred["probabilities"])


# ---------------------------------------------------------------------------
# Tab 2: Integrated Gradients (Section 4.7.2)
# Calls FinBERTIGExplainer.explain() and renders the colour-coded HTML
# attribution visualisation. IG runs in ~1s on CPU (n_steps=30 forward/backward
# passes through the model). Green = supports predicted class, red = opposes.
# ---------------------------------------------------------------------------
with tab_ig:
    st.subheader("Integrated Gradients")
    st.write("Green tokens support the explained class; red tokens oppose it.")

    if st.button("Explain with IG"):
        out = ig_explainer.explain(text)

        st.markdown(
            f"**Explaining:** `{out['target_explained']['label'].upper()}` "
            f"(predicted: `{out['predicted']['label'].upper()}` | confidence: `{out['predicted']['confidence']:.3f}`)"
        )

        # HTML attribution string rendered natively via st.markdown unsafe_allow_html
        # (Section 4.7.5 — no frontend build step required)
        st.markdown(out["html"], unsafe_allow_html=True)

        # Ranked attribution table — top 12 by absolute value for the ranked display
        pairs = [(t, a) for t, a in zip(out["tokens"], out["attributions"]) if str(t).strip() != ""]
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:12]
        st.write("Top tokens by |attribution|:")
        st.table([{"token": t, "attribution": float(a)} for t, a in pairs_sorted])


# ---------------------------------------------------------------------------
# Tab 3: SHAP (Section 4.7.3)
# Calls FinBERTSHAPExplainer.explain() using the partition algorithm with a
# tokeniser-aware masker (Section 4.5.2). Slower than IG (~8-15s on CPU)
# due to coalition sampling (max_evals=250 forward passes). Uses the same
# HTML colour scheme as the IG tab for direct visual comparison.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Tab 4: Batch CSV (Section 4.7.4)
# Accepts a CSV upload, auto-detects the text column, runs bulk FinBERT
# prediction on all rows, and optionally appends top-k IG tokens per row.
# IG explanation is optional and row-capped because it adds ~1-3s per sentence
# on CPU — satisfies NF1 (inference latency) and NF6 (scalability, Section 3.4.2).
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch CSV Upload")
    st.write("Upload a CSV of headlines/sentences and get predictions + confidence, plus Top-IG tokens for explanations.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

        st.write("Preview:")
        st.dataframe(df.head(10), use_container_width=True)

        # Auto-detect text column; fall back to manual selection if not found
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

        # IG explanation controls — optional because IG adds per-row latency
        explain_ig = st.checkbox("Add Top IG tokens to output (slower)", value=True)
        top_k = st.slider("Top-K IG tokens", min_value=3, max_value=12, value=5)
        max_explain = st.number_input(
            "Max rows to explain with IG (speed control)",
            min_value=1,
            max_value=int(max_rows),
            value=min(int(max_rows), 100),
        )

        if st.button("Run batch predictions", type="primary"):
            # Step 1: run FinBERT on all rows (fast — single batched forward pass)
            results = run_batch_predictions(df_run, text_col=text_col, predictor=predictor)

            # Step 2: optionally append top-k IG tokens (slower — one explain() call per row)
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

            # Summary visualisations
            c1, c2 = st.columns(2)
            with c1:
                st.write("Sentiment distribution:")
                st.bar_chart(results["sentiment"].value_counts())
            with c2:
                st.write("Confidence summary:")
                st.write(results["confidence"].describe())

            st.write("Results:")
            st.dataframe(results, use_container_width=True)

            # Downloadable CSV output (FR12 — Could Have requirement, Section 3.4.1)
            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                data=csv_bytes,
                file_name="batch_sentiment_results.csv",
                mime="text/csv",
            )
        
