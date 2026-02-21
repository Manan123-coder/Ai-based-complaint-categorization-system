# app.py - Beautiful Dashboard (Fixed Version)
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="AI Complaint Categorizer",
    page_icon="🤖",
    layout="wide"
)

# Load Model & Data
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
df = pd.read_csv('dataset.csv')

# Get Category Stats
category_counts = df['category'].value_counts().reset_index()
category_counts.columns = ['Category', 'Count']

# Initialize Session State
if 'prediction_logs' not in st.session_state:
    st.session_state.prediction_logs = []

# Prediction Function
def predict_category(complaint_text):
    from preprocess import preprocess_text
    processed = preprocess_text(complaint_text)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    return prediction

# =======================
# SIDEBAR
# =======================
with st.sidebar:
    st.title("🤖 AI Complaint System")
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("Total Complaints", len(df))
    st.metric("Categories", 5)
    st.markdown("---")
    st.markdown("### 📈 Categories")
    for idx, row in category_counts.iterrows():
        st.write(f"{row['Category']}: **{row['Count']}**")

# =======================
# MAIN CONTENT
# =======================

st.title("🎯 AI Complaint Categorization")
st.markdown("### Intelligent NLP-Powered Classification")

# Input Section
st.markdown("## 📝 Enter Your Complaint")
complaint_input = st.text_area(
    "Describe your issue:",
    placeholder="e.g., My app keeps crashing...",
    height=80
)

if st.button("🔍 Predict Category"):
    if complaint_input.strip():
        category = predict_category(complaint_input)
        st.success(f"**Predicted Category: {category}**")
        
        st.session_state.prediction_logs.append({
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Complaint': complaint_input[:40] + "...",
            'Category': category
        })
    else:
        st.error("Please enter a complaint!")

# Analytics Dashboard
st.markdown("---")
st.markdown("## 📊 Analytics Dashboard")

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("📁 Total Data", len(df))
m2.metric("🏷️ Categories", 5)
m3.metric("🔮 Predictions", len(st.session_state.prediction_logs))
m4.metric("🔥 Top Category", category_counts.iloc[0]['Category'])

# Charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### 📊 Bar Chart")
    fig_bar = px.bar(
        category_counts,
        x='Category',
        y='Count',
        color='Category',
        template="plotly_white"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    st.markdown("### 🥧 Pie Chart")
    fig_pie = px.pie(
        category_counts,
        values='Count',
        names='Category',
        hole=0.4,
        template="plotly_white"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Prediction Logs
st.markdown("---")
st.markdown("## 📋 Recent Predictions")

if st.session_state.prediction_logs:
    logs_df = pd.DataFrame(st.session_state.prediction_logs[-10:])
    st.dataframe(logs_df, use_container_width=True, hide_index=True)
else:
    st.info("No predictions yet!")

# Footer
st.markdown("---")
st.markdown("Made by Manan using Streamlit")