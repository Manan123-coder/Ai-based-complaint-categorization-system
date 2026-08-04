import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime
import os

st.set_page_config(
    page_title="AI Complaint Categorizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

LIVE_FILE = 'live_complaints.csv'

@st.cache_data(ttl=2)  
def load_fresh_data():
    """Loads FRESHEST data every time"""
    df_original = pd.read_csv('dataset.csv')
    
    if os.path.exists(LIVE_FILE):
        try:
            live_df = pd.read_csv(LIVE_FILE)
        except:
            live_df = pd.DataFrame(columns=['timestamp', 'complaint_text', 'predicted_category'])
    else:
        live_df = pd.DataFrame(columns=['timestamp', 'complaint_text', 'predicted_category'])
    
    return df_original, live_df

df_original, live_complaints_df = load_fresh_data()

def get_combined_dataset(df_orig, df_live):
    if len(df_live) == 0:
        return df_orig.copy()
    
    live_renamed = df_live[['complaint_text', 'predicted_category']].rename(
        columns={'predicted_category': 'category'}
    )
    
    combined = pd.concat([df_orig, live_renamed], ignore_index=True)
    return combined

combined_df = get_combined_dataset(df_original, live_complaints_df)
category_counts = combined_df['category'].value_counts().reset_index()
category_counts.columns = ['Category', 'Count']

if 'prediction_logs' not in st.session_state:
    st.session_state.prediction_logs = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = len(live_complaints_df)

def predict_category(complaint_text):
    from preprocess import preprocess_text
    processed = preprocess_text(complaint_text)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    return prediction

def save_prediction(complaint_text, category):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = pd.DataFrame({
            'timestamp': [timestamp],
            'complaint_text': [complaint_text],
            'predicted_category': [category]
        })
        
        if os.path.exists(LIVE_FILE):
            df_live = pd.read_csv(LIVE_FILE)
            updated_df = pd.concat([df_live, new_row], ignore_index=True)
        else:
            updated_df = new_row
        updated_df.to_csv(LIVE_FILE, index=False)
        return True
    except PermissionError:
        st.error("❌ Close Excel! File is locked.")
        return False
    except Exception as e:
        st.error(f"Save error: {e}")
        return False

with st.sidebar:
    st.title("🤖 AI Complaint System")
    st.markdown("---")
        
    col1, col2, col3 = st.columns(3)
    col1.metric("Original", len(df_original))
    col2.metric(" Live Predictions", len(live_complaints_df))
    col3.metric(" Total", len(combined_df))
    
    st.markdown("---")
    st.markdown("###  Live Categories")
    for idx, row in category_counts.iterrows():
        st.write(f"{row['Category']}: **{int(row['Count'])}**")
    
    st.markdown("---")
    st.markdown("###  Refresh Controls")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button(" Refresh Data", type="secondary"):
            st.cache_data.clear()
            st.rerun()
    
    with col_btn2:
        if st.button(" Clear Live CSV", type="secondary"):
            if os.path.exists(LIVE_FILE):
                os.remove(LIVE_FILE)
                st.success(" Live CSV cleared!")
                st.rerun()
            else:
                st.info("No live CSV to clear")

# MAIN CONTENT
st.title(" AI Complaint Categorization")
st.markdown("### Intelligent NLP-Powered Classification")

st.markdown("##  Enter Your Complaint")
complaint_input = st.text_area(
    "Describe your issue:",
    placeholder="e.g., My app keeps crashing...",
    height=80
)

if st.button("Predict Category", type="secondary"):
    if complaint_input.strip():
        category = predict_category(complaint_input)
        st.success(f"**Predicted Category: {category}**")
        
        if save_prediction(complaint_input, category):
            st.info(" Saved to live_complaints.csv!")
            st.session_state.prediction_logs.append({
                'Time': datetime.now().strftime("%H:%M:%S"),
                'Complaint': complaint_input[:40] + "...",
                'Category': category
            })
            st.session_state.total_predictions += 1
            st.rerun()
    else:
        st.error("Please enter a complaint!")

# Analytics Dashboard
st.markdown("---")
st.markdown("##  Analytics Dashboard")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Original Data", len(df_original))
m2.metric("Live Predictions", len(live_complaints_df))
m3.metric("Total Dataset", len(combined_df))
m4.metric("Top Category", category_counts.iloc[0]['Category'])

# Charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("###Bar Chart")
    fig_bar = px.bar(
        category_counts,
        x='Category', y='Count',
        color='Category',
        template="plotly_white",
        title=f"Total: {len(combined_df)} complaints"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    st.markdown("### Pie Chart")
    fig_pie = px.pie(
        category_counts,
        values='Count', names='Category',
        hole=0.4,
        template="plotly_white",
        title=f"Total: {len(combined_df)}"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Logs
st.markdown("---")
st.markdown("## Recent Predictions")
if st.session_state.prediction_logs:
    logs_df = pd.DataFrame(st.session_state.prediction_logs[-10:])
    st.dataframe(logs_df, use_container_width=True, hide_index=True)
else:
    st.info("No predictions yet!")

# Download
if os.path.exists(LIVE_FILE):
    with open(LIVE_FILE, 'r') as f:
        csv_data = f.read()
    st.download_button(
        "Download Live CSV",
        csv_data, "live_complaints.csv", "text/csv"
    )

st.markdown("---")
st.markdown("Made by Manan using Streamlit")
