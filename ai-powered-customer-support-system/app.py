import streamlit as st
import pandas as pd
import os
import csv
from datetime import datetime
import uuid

from src.inferencial import predict
from src.priority import detect_priority
from src.xai import explain_prediction

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Customer Support Intelligence",
    page_icon="📩",
    layout="centered"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
# 📩 AI Customer Support Intelligence System  
### 🤖 DistilBERT • Multi-Label NLP   

Turning customer messages into **actionable support tickets**
""")

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("⚙️ Control Panel")
is_admin = st.sidebar.checkbox("🔐 Admin Mode")

st.sidebar.markdown("""
**Features**
- 🧠 Multi-label NLP  
- 🚦 Priority Detection  
- 🆔 Ticket Tracking  
- 📊 Analytics Dashboard  
- 💡 Suggested Actions  
""")

# --------------------------------------------------
# TICKET LOG SETUP
# --------------------------------------------------
LOG_FILE = "ticket_log.csv"

def log_ticket(text, issues, priority, confidence):
    file_exists = os.path.exists(LOG_FILE)
    ticket_id = f"TCKT-{uuid.uuid4().hex[:6].upper()}"

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
       
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow([
                "ticket_id", "timestamp", "message",
                "issues", "priority", "confidence", "status"
            ])

        writer.writerow([
            ticket_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            text,
            "; ".join(issues),
            priority,
            round(confidence, 2),
            "Open"
        ])
    return ticket_id

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
st.subheader("✍️ Enter Customer Message")

text = st.text_area(
    "",
    placeholder="e.g. Payment failed and app not working...",
    height=120
)

# --------------------------------------------------
# SAMPLE MESSAGES
# --------------------------------------------------
st.markdown("### 🧪 Try Sample Messages")

c1, c2, c3 = st.columns(3)

if c1.button("💳 Payment Issue"):
    text = "Payment failed and app not working"

if c2.button("🚚 Delivery Delay"):
    text = "My order is delayed and no response from support"

if c3.button("😊 Feedback"):
    text = "Thanks, the new update works great"

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------
st.divider()

if st.button("🔍 Analyze Message"):
    if not text.strip():
        st.warning("⚠️ Please enter a message")
    else:
        with st.spinner("🤖 Analyzing..."):
            issues, confidence = predict(text)
            priority = detect_priority(text)
            keywords = explain_prediction(text)

            ticket_id = log_ticket(text, issues, priority, confidence)

        st.success(f"✅ Analysis Complete | Ticket ID: {ticket_id}")

        # ---------------- RESULTS ----------------
        st.subheader("📌 Detected Issues")
        for issue in issues:
            st.success(issue)

        # ---------------- PRIORITY ----------------
        st.subheader("🚦 Priority Level")
        if priority == "High":
            st.error("🔴 High Priority")
        elif priority == "Medium":
            st.warning("🟠 Medium Priority")
        else:
            st.success("🟢 Low Priority")

        # ---------------- CONFIDENCE ----------------
        st.subheader("📊 Prediction Confidence")
        st.progress(min(confidence, 1.0))
        st.caption(f"Confidence score: **{confidence:.2f}**")

        # ---------------- SUGGESTED ACTION ----------------
        st.subheader("💡 Suggested Action")
        if "Billing & Payment Issue" in issues:
            st.info("Forward to **Billing Team**")
        elif "Technical Support Issue" in issues:
            st.info("Assign to **Technical Support Team**")
        elif "Delivery / Service Delay" in issues:
            st.info("Escalate to **Logistics Team**")
        elif "Feedback / Praise" in issues:
            st.info("Tag as **Positive Feedback**")
        else:
            st.info("No immediate action required")

        # ---------------- EXPLAINABILITY ----------------
        st.subheader("🧠 Why this prediction?")
        if keywords:
            st.info("🔑 Important words: " + ", ".join(keywords))
        else:
            st.info("No strong keywords detected")

# --------------------------------------------------
# ADMIN DASHBOARD
# --------------------------------------------------
if is_admin:
    st.divider()
    st.subheader("📊 Admin Analytics Dashboard")

    if not os.path.exists(LOG_FILE):
        st.info("No tickets logged yet.")
    else:
        df = pd.read_csv(LOG_FILE, on_bad_lines='skip', engine='python')
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        st.markdown("### 🔥 Most Reported Issues")
        st.bar_chart(df["issues"].str.split("; ").explode().value_counts())

        st.markdown("### 🚦 Priority Distribution")
        st.bar_chart(df["priority"].value_counts())

        st.markdown("### 📈 Daily Ticket Trend")
        st.line_chart(df.groupby("date").size())

        st.markdown("### 🕒 Recent Tickets")
        st.dataframe(df.tail(10), use_container_width=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption("""
🛠 Optimized for **low-resource systems (4GB RAM)**  
🎓 Industry-style ML project with tracking & analytics
""")
