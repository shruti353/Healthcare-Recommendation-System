import streamlit as st
import pandas as pd
import numpy as np
import fitz  # pymupdf
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import google.generativeai as genai

# Configure the Gemini API
# The API key is loaded from Streamlit secrets, which is a secure way to handle it.
# For local testing, ensure you have a .streamlit/secrets.toml file with GOOGLE_API_KEY = "YOUR_API_KEY"
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Initialize the Generative Model. Using 'gemini-2.0-flash' for text generation.
model = genai.GenerativeModel('gemini-2.0-flash')

# --------------------- PDF to Text ---------------------
def extract_text_from_pdf(file):
    """
    Extracts text content from an uploaded PDF file.

    Args:
        file: A file-like object representing the PDF.

    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    try:
        # Open the PDF file using fitz (PyMuPDF)
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            # Iterate through each page and extract text
            for page in doc:
                text += page.get_text()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        text = "Could not extract text from PDF."
    return text

# --------------------- Excel/CSV Loader ---------------------
def load_health_data(file):
    """
    Loads health data from an uploaded CSV or Excel file into a Pandas DataFrame.
    Fills any NaN values with 0.

    Args:
        file: A file-like object representing the CSV or Excel file.

    Returns:
        pd.DataFrame: The loaded health data.
    """
    df = pd.DataFrame() # Initialize an empty DataFrame
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else: # Assumes .xlsx if not .csv
            df = pd.read_excel(file)
        df.fillna(0, inplace=True) # Fill missing values with 0
    except Exception as e:
        st.error(f"Error loading health data from file: {e}")
    return df

# --------------------- Health Recommendation Engine ---------------------
def generate_recommendations(data):
    """
    Generates health recommendations based on extracted text (symptoms)
    or numerical health data (vitals).

    Args:
        data (str or pd.DataFrame): Either text containing symptoms or
                                    a DataFrame with health vitals.

    Returns:
        list: A list of health recommendations.
    """
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
    else: # Assumes data is a Pandas DataFrame
        # Check for relevant columns and generate recommendations based on thresholds
        if "BMI" in data.columns and not data.empty and data['BMI'].iloc[0] > 25:
            recs.append("You're overweight. Consider a fitness plan.")
        if "Blood Pressure" in data.columns and not data.empty and data['Blood Pressure'].iloc[0] > 130:
            recs.append("Control salt intake and stress.")
        if "Glucose" in data.columns and not data.empty and data['Glucose'].iloc[0] > 140:
            recs.append("Monitor sugar and consult for diabetes.")
        if "Cholesterol" in data.columns and not data.empty and data['Cholesterol'].iloc[0] > 200:
            recs.append("Avoid fried food and check lipids.")
        if not recs:
            recs.append("Healthy vitals detected. Keep it up!")
    return recs

# --------------------- Risk Prediction ---------------------
def predict_risks(df):
    """
    Predicts health risks based on numerical health data using a RandomForestClassifier.

    Args:
        df (pd.DataFrame): DataFrame containing health vitals.

    Returns:
        dict: A dictionary containing risk level and confidence.
    """
    risks = {}
    required_cols = ["Age", "BMI", "Blood Pressure", "Glucose", "Cholesterol"]

    # Check if all required columns are present and DataFrame is not empty
    if not all(col in df.columns for col in required_cols) or df.empty:
        st.warning("Missing required columns for risk prediction or no data available.")
        return risks

    try:
        # Prepare features (X) and target (y)
        X = df[required_cols].values
        # Define a simple target: 1 if high blood pressure or high glucose, else 0
        y = np.array([1 if x[2] > 140 or x[3] > 150 else 0 for x in X])

        # Initialize and train the RandomForestClassifier model
        model = RandomForestClassifier(random_state=42) # Added random_state for reproducibility
        model.fit(X, y)

        # Make predictions and get probabilities for the first entry
        pred = model.predict(X)
        prob = model.predict_proba(X)

        # Determine risk level and confidence
        risks["Risk Level"] = "High" if pred[0] == 1 else "Low"
        # Ensure index exists before accessing
        if len(prob[0]) > pred[0]:
            risks["Confidence"] = f"{int(prob[0][pred[0]] * 100)}%"
        else:
            risks["Confidence"] = "N/A" # Handle case where prediction index is out of bounds

    except Exception as e:
        st.error(f"Error during risk prediction: {e}")
    return risks

# --------------------- Visual Analytics ---------------------
def show_charts(df):
    """
    Displays interactive histograms for key health metrics using Plotly Express.

    Args:
        df (pd.DataFrame): DataFrame containing health data.
    """
    st.subheader("📈 Visual Health Analytics")
    chart_cols = ["Age", "BMI", "Blood Pressure", "Glucose", "Cholesterol"]
    for col in chart_cols:
        if col in df.columns and not df.empty:
            try:
                fig = px.histogram(df, x=col, nbins=10, title=f"{col} Distribution")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate chart for {col}: {e}")

# --------------------- Chatbot Function ---------------------
def chatbot_response(query):
    """
    Generates a health advice response using the Gemini API.

    Args:
        query (str): The user's health-related question.

    Returns:
        str: The AI's response or an error message.
    """
    try:
        # Generate content using the Gemini model with the user's query
        response = model.generate_content(query)
        reply = response.text
    except Exception as e:
        # Catch any exceptions during API call and return an error message
        reply = f"Error: Could not get a response from the AI. {str(e)}"
    return reply

# --------------------- Streamlit App Layout ---------------------
st.set_page_config(page_title="🩺 Mega Health Recommender", layout="wide")
st.title("🧬 Mega Health Recommendation System")
st.markdown("Upload your health reports and get personalized, AI-powered suggestions 🚀")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["📤 Upload Report", "🤖 Ask AI HealthBot"])

with tab1:
    uploaded_file = st.file_uploader("Upload .xlsx, .csv or .pdf", type=["xlsx", "csv", "pdf"])
    if uploaded_file:
        if uploaded_file.name.endswith("pdf"):
            text = extract_text_from_pdf(uploaded_file)
            st.subheader("📄 Extracted Text from PDF:")
            st.text(text)
            recommendations = generate_recommendations(text)
            risks = {} # No numerical data for risk prediction from PDF text
        else: # Assumes Excel or CSV
            df = load_health_data(uploaded_file)
            st.subheader("📊 Uploaded Health Data:")
            st.dataframe(df)
            recommendations = generate_recommendations(df)
            risks = predict_risks(df)
            show_charts(df)

        # Display risk assessment if available
        if risks:
            st.subheader("🔬 Risk Assessment")
            for k, v in risks.items():
                st.write(f"**{k}**: {v}")

        # Display personalized recommendations
        st.subheader("✅ Personalized Recommendations")
        if recommendations:
            for r in recommendations:
                st.markdown(f"- {r}")
        else:
            st.info("No specific recommendations generated based on the provided data.")

with tab2:
    st.subheader("🤖 Ask your Health-related Question")

    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Use a form for input and submission. clear_on_submit=True clears the text input.
    with st.form("chat_form", clear_on_submit=True):
        # Text input for user's query, now inside the form
        user_input = st.text_input("Type your health query here...", key="chat_input")
        # Submit button for the form
        submitted = st.form_submit_button("Send")

        # If the form is submitted and there's user input
        if submitted and user_input:
            # Clear chat history before appending new question/answer
            st.session_state.chat_history = [] 

            # Call the chatbot function
            reply = chatbot_response(user_input)
            
            # Append user's query and AI's reply to chat history
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("HealthBot", reply))
            
            # No need for st.session_state.input_key = "" or st.experimental_rerun() here
            # because clear_on_submit=True handles clearing the input,
            # and Streamlit will rerun naturally to display updated chat_history.

    # Display chat history
    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")
