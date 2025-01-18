import nltk
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import re
import streamlit as st
from io import StringIO
from PyPDF2 import PdfReader
from docx import Document

# Set custom NLTK data path
nltk.data.path.append(r'C:\Users\Y.Veeranaganjineya\AppData\Roaming\nltk_data')

# Download necessary NLTK resources to the specified path
nltk.download('punkt', download_dir=r'C:\Users\Y.Veeranaganjineya\AppData\Roaming\nltk_data')

# Load HuggingFace summarization pipeline (abstractive summarization)
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except ImportError:
    summarizer = None
    st.warning("Abstractive summarization unavailable. Please install PyTorch or TensorFlow.")

# Function to clean and preprocess the text
def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9.,!?\'\s]", "", text)  # Remove unwanted characters
    text = text.strip()
    return text

# Function to perform extractive summarization
def extractive_summary(text):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:3])  # Use the first 3 sentences for simplicity

# Function to perform abstractive summarization
def abstractive_summary(text):
    if not summarizer:
        return "Abstractive summarization is unavailable. Please install PyTorch or TensorFlow."
    try:
        summarized = summarizer(text, max_length=200, min_length=50, do_sample=False)
        return summarized[0]["summary_text"]
    except Exception as e:
        return f"Error in abstractive summarization: {e}"

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

# Function to extract text from DOCX
def extract_text_from_docx(file):
    try:
        doc = Document(file)
        text = " ".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error reading DOCX: {e}"

# Main function to summarize the report
def summarize_report(text):
    cleaned_text = clean_text(text)
    extractive = extractive_summary(cleaned_text)
    abstractive = abstractive_summary(cleaned_text)
    return extractive, abstractive

# Streamlit UI
def app():
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(to right, #ece9e6, #ffffff);
            padding: 10px;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(to bottom, #ffffff, #d9e4f5);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Multiple Document and Plain Text Summarizer")
    st.markdown("Upload multiple doctor's reports (PDF/DOCX) or enter plain text to generate individual summaries.")

    uploaded_files = st.file_uploader("Choose files (PDF/DOCX):", type=["pdf", "docx"], accept_multiple_files=True)
    plain_text = st.text_area("Or enter plain text here:")

    if not uploaded_files and not plain_text.strip():
        st.warning("Summerize The Text")
        return

    if plain_text.strip():
        st.subheader("Processing Plain Text")
        extractive, abstractive = summarize_report(plain_text.strip())

        st.subheader("Extractive Summary of Plain Text")
        st.write(extractive)

        st.subheader("Abstractive Summary of Plain Text")
        st.write(abstractive)

    for uploaded_file in uploaded_files:
        st.subheader(f"Processing file: {uploaded_file.name}")

        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            continue

        if isinstance(text, str) and text.startswith("Error"):
            st.error(f"Error in file {uploaded_file.name}: {text}")
            continue

        st.subheader(f"Input Report Text from {uploaded_file.name}")
        st.write(text)

        extractive, abstractive = summarize_report(text)

        st.subheader(f"Extractive Summary of {uploaded_file.name}")
        st.write(extractive)

        st.subheader(f"Abstractive Summary of {uploaded_file.name}")
        st.write(abstractive)

# Run the Streamlit app
if __name__ == "__main__":
    app()
