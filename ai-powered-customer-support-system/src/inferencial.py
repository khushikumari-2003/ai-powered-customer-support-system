import torch
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# --------------------------------------------------
# MODEL CONFIG
# --------------------------------------------------
MODEL_PATH = "src/models/bert_multilabel"

LABELS = [
    "Billing & Payment Issue",
    "Technical Support Issue",
    "Account / Login Issue",
    "Product Quality Issue",
    "Delivery / Service Delay",
    "Feedback / Praise",
    "Spam / Irrelevant"
]

# --------------------------------------------------
# LOAD MODEL (CACHED – 4GB RAM SAFE)
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    model.to("cpu")
    return tokenizer, model

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------
def predict(text: str):
    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.sigmoid(logits)[0].tolist()

    # -------- ML-BASED PREDICTION --------
    issues = [LABELS[i] for i, p in enumerate(probs) if p >= 0.35]

    t = text.lower()

    # -------- STRONG RULE OVERRIDES (INDUSTRY FIX) --------
    if any(w in t for w in ["payment", "refund", "charged"]):
        issues.append("Billing & Payment Issue")

    if any(w in t for w in ["error", "failed", "not working", "crash", "bug"]):
        issues.append("Technical Support Issue")

    if any(w in t for w in ["login", "account", "password"]):
        issues.append("Account / Login Issue")

    if any(w in t for w in ["delay", "late", "delivery"]):
        issues.append("Delivery / Service Delay")

    if any(w in t for w in ["broken", "defective", "quality"]):
        issues.append("Product Quality Issue")

    if any(w in t for w in ["thank", "thanks", "great", "good", "love"]):
        issues.append("Feedback / Praise")

    # Remove duplicates
    issues = list(set(issues))

    # -------- FINAL FALLBACK --------
    if not issues:
        issues = ["Spam / Irrelevant"]

    confidence = max(probs)

    return issues, confidence



