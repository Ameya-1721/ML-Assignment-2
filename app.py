from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    ConfusionMatrixDisplay, classification_report,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TEST_DATA_PATH = BASE_DIR / "test_data.csv"
METRICS_PATH = BASE_DIR / "model_metrics.csv"
METADATA_PATH = BASE_DIR / "model_metadata.json"

st.set_page_config(page_title="Shopper Purchase Intention", page_icon="🛍️", layout="wide")

@st.cache_resource
def load_artifacts():
    if not METADATA_PATH.exists():
        raise FileNotFoundError("model_metadata.json is missing. Run the assignment notebook first.")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    models = {}
    missing = []
    for key, filename in metadata["model_files"].items():
        path = MODEL_DIR / filename
        if path.exists():
            models[key] = joblib.load(path)
        else:
            missing.append(str(path.name))
    if missing:
        raise FileNotFoundError("Missing trained model file(s): " + ", ".join(missing) + ". Run the notebook export cell again.")
    metrics = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()
    return metadata, models, metrics


def normalize_target(values):
    """Convert common boolean/string/numeric Revenue encodings to 0/1."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(int)
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any() or not set(numeric.unique()).issubset({0, 1}):
            raise ValueError("Revenue must contain only 0/1 or True/False values.")
        return numeric.astype(int)
    mapping = {
        "true": 1, "false": 0, "1": 1, "0": 0,
        "yes": 1, "no": 0, "purchase": 1, "no purchase": 0,
    }
    normalized = values.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        bad = values[normalized.isna()].astype(str).unique()[:5]
        raise ValueError(f"Unrecognized Revenue value(s): {bad}. Use True/False or 1/0.")
    return normalized.astype(int)


def get_scores(estimator, features):
    """Return a continuous score for the positive class (Revenue=1)."""
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)
        classes = list(estimator.classes_)
        return probabilities[:, classes.index(1)]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(features)
        classes = list(estimator.classes_)
        # Binary sklearn decision_function is oriented toward classes_[1].
        return scores if classes.index(1) == 1 else -scores
    return estimator.predict(features).astype(float)


st.title("🛍️ Online Shopper Purchase Intention")
st.caption("Binary classification of e-commerce sessions using five machine-learning classifiers")
st.markdown(
    "This application evaluates saved models on an uploaded labelled test CSV. "
    "It **does not retrain** models on uploaded data. The positive class is `Revenue = 1` (purchase)."
)

try:
    metadata, models, saved_metrics = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.info("Place the notebook-generated `model/`, `model_metadata.json`, `model_metrics.csv`, and `test_data.csv` beside this app.py file.")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Dataset sessions", f"{metadata['dataset_rows']:,}")
m2.metric("Model input features", str(len(metadata["feature_columns"])))
m3.metric("Positive class", "Purchase (Revenue = 1)")

with st.expander("About the dataset and methodology"):
    st.write(metadata.get("dataset_description", "UCI Online Shoppers Purchasing Intention Dataset"))
    st.write("The original `PageValues` field is excluded from model inputs as a leakage-aware choice because it is a conversion-related analytics measure.")
    st.write("All five pipelines were fitted on the training split only. The uploaded CSV is used only for evaluation.")

st.subheader("1. Upload labelled test data")
uploaded_file = st.file_uploader("Upload a test CSV (must include all model input columns and Revenue)", type=["csv"])
if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)
        input_source = f"Uploaded file: {uploaded_file.name}"
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
else:
    if not TEST_DATA_PATH.exists():
        st.warning("No CSV uploaded and default test_data.csv is missing. Upload the test CSV exported by the notebook.")
        st.stop()
    test_data = pd.read_csv(TEST_DATA_PATH)
    input_source = "Notebook-generated holdout test_data.csv"

st.caption(input_source)
feature_columns = metadata["feature_columns"]
required_columns = feature_columns + [metadata["target_column"]]
missing_columns = [col for col in required_columns if col not in test_data.columns]
if missing_columns:
    st.error("The CSV is missing required column(s): " + ", ".join(missing_columns))
    st.stop()

try:
    X_test = test_data[feature_columns].copy()
    y_test = normalize_target(test_data[metadata["target_column"]])
except Exception as exc:
    st.error(f"Input validation failed: {exc}")
    st.stop()

if len(test_data) < 2:
    st.error("Please upload at least two labelled test rows.")
    st.stop()

with st.expander("Preview uploaded test data"):
    st.dataframe(test_data.head(20), use_container_width=True)
    st.write(f"Rows: {len(test_data):,} | Purchase sessions: {int(y_test.sum()):,} | Non-purchase sessions: {int((y_test == 0).sum()):,}")

st.subheader("2. Select a classifier")
model_names = list(models.keys())
selected_model = st.selectbox("Model", model_names, index=0)
estimator = models[selected_model]
y_pred = estimator.predict(X_test).astype(int)
y_scores = get_scores(estimator, X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y_test, y_pred),
}
try:
    metrics["ROC-AUC"] = roc_auc_score(y_test, y_scores) if y_test.nunique() == 2 else np.nan
except Exception:
    metrics["ROC-AUC"] = np.nan

st.subheader("3. Evaluation metrics")
metric_order = ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1 Score", "MCC"]
cols = st.columns(6)
for col, name in zip(cols, metric_order):
    value = metrics.get(name, np.nan)
    col.metric(name, "N/A" if pd.isna(value) else f"{value:.4f}")
if pd.isna(metrics["ROC-AUC"]):
    st.warning("ROC-AUC requires both purchase and non-purchase examples in the uploaded test file.")

st.subheader("4. Confusion matrix and classification report")
left, right = st.columns([1, 1])
with left:
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No purchase", "Purchase"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(f"{selected_model} — Confusion Matrix")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
with right:
    report = classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["No purchase", "Purchase"],
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)

st.subheader("5. Saved holdout comparison")
if not saved_metrics.empty:
    st.dataframe(saved_metrics.round(4), use_container_width=True)
    st.caption("These are the notebook's original holdout metrics. The metrics above are recalculated on the currently uploaded CSV.")

with st.expander("Download predictions for this evaluation"):
    output = test_data.copy()
    output["Actual_Revenue"] = y_test.to_numpy()
    output["Predicted_Revenue"] = y_pred
    output["Purchase_Score"] = y_scores
    st.download_button(
        "Download predictions CSV",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="shopper_purchase_predictions.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Academic assignment demonstration · UCI Online Shoppers Purchasing Intention Dataset · Models are evaluated only; uploaded data is never used for fitting.")
