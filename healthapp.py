# mega_health_app.py
import streamlit as st
import pandas as pd
import numpy as np
import fitz  # pymupdf
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# --------------------- PDF to Text ---------------------
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# --------------------- Excel/CSV Loader ---------------------
def load_health_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.fillna(0, inplace=True)
    return df

# --------------------- Health Recommendation Engine ---------------------
def generate_recommendations(data):
    recs = []

    if isinstance(data, str):
        text = data.lower()
        if "fatigue" in text:
            recs.append("Check for iron deficiency.")
        if "headache" in text:
            recs.append("Stay hydrated and get adequate sleep.")
        if "chest pain" in text:
            recs.append("Seek immediate medical attention.")
        if "stress" in text:
            recs.append("Try yoga and mindfulness meditation.")
        if not recs:
            recs.append("No direct symptoms found. Stay proactive.")
    else:
        if "BMI" in data.columns and data['BMI'].iloc[0] > 25:
            recs.append("You're overweight. Consider a fitness plan.")
        if "Blood Pressure" in data.columns and data['Blood Pressure'].iloc[0] > 130:
            recs.append("Control salt intake and stress.")
        if "Glucose" in data.columns and data['Glucose'].iloc[0] > 140:
            recs.append("Monitor sugar and consult for diabetes.")
        if "Cholesterol" in data.columns and data['Cholesterol'].iloc[0] > 200:
            recs.append("Avoid fried food and check lipids.")
        if not recs:
            recs.append("Healthy vitals detected. Keep it up!")
    return recs

# --------------------- Risk Prediction ---------------------
def predict_risks(df):
    risks = {}
    required = {"Age", "BMI", "Blood Pressure", "Glucose", "Cholesterol"}
    if not required.issubset(df.columns):
        return risks

    X = df[["Age", "BMI", "Blood Pressure", "Glucose", "Cholesterol"]].values
    y = np.array([1 if x[2] > 140 or x[3] > 150 else 0 for x in X])

    model = RandomForestClassifier()
    model.fit(X, y)
    pred = model.predict(X)
    prob = model.predict_proba(X)

    risks["Risk Level"] = "High" if pred[0] == 1 else "Low"
    risks["Confidence"] = f"{int(prob[0][pred[0]] * 100)}%"
    return risks

# --------------------- Visual Analytics ---------------------
def show_charts(df):
    st.subheader("📈 Visual Health Analytics")
    chart_cols = ["Age", "BMI", "Blood Pressure", "Glucose", "Cholesterol"]
    for col in chart_cols:
        if col in df.columns:
            fig = px.histogram(df, x=col, nbins=10, title=f"{col} Distribution")
            st.plotly_chart(fig, use_container_width=True)

# --------------------- Chatbot ---------------------
import openai
import os

# Load API key from Streamlit secrets
openai.api_key = os.getenv("OPENAI_API_KEY")

# GPT Chatbot Function
def chatbot_response(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful healthcare assistant. Answer factually, but don't diagnose or give medical advice."
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"


# --------------------- Streamlit App ---------------------
st.set_page_config(page_title="🩺 Mega Health Recommender", layout="wide")
st.title("🧬 Mega Health Recommendation System")
st.markdown("Upload your health reports and get personalized, AI-powered suggestions 🚀")

tab1, tab2 = st.tabs(["📤 Upload Report", "🤖 Ask AI HealthBot"])

with tab1:
    uploaded_file = st.file_uploader("Upload .xlsx, .csv or .pdf", type=["xlsx", "csv", "pdf"])
    if uploaded_file:
        if uploaded_file.name.endswith("pdf"):
            text = extract_text_from_pdf(uploaded_file)
            st.subheader("📄 Extracted Text from PDF:")
            st.text(text)
            recommendations = generate_recommendations(text)
            risks = {}
        else:
            df = load_health_data(uploaded_file)
            st.subheader("📊 Uploaded Health Data:")
            st.dataframe(df)
            recommendations = generate_recommendations(df)
            risks = predict_risks(df)
            show_charts(df)

        if risks:
            st.subheader("🔬 Risk Assessment")
            for k, v in risks.items():
                st.write(f"**{k}**: {v}")

        st.subheader("✅ Personalized Recommendations")
        for r in recommendations:
            st.markdown(f"- {r}")

with tab2:
    st.subheader("🤖 Ask your Health-related Question")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Type your health query here...")

    if user_input:
        reply = chatbot_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("HealthBot", reply))

    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")
