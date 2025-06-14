
# main_app.py
import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdfplumber
import pytesseract
# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import Image
import docx
import together
import io
import easyocr
reader = easyocr.Reader(['en'], gpu=False)

# Set Together API Key
# together.api_key = st.secrets["TOGETHER_API_KEY"] if "TOGETHER_API_KEY" in st.secrets else st.text_input("Enter Together API Key", type="password")

# Try to use secrets.toml, else ask user
if "TOGETHER_API_KEY" in st.secrets:
    together.api_key = st.secrets["TOGETHER_API_KEY"]
else:
    user_key = st.text_input("🔑 Enter Together API Key", type="password")
    if user_key:
        together.api_key = user_key

# Helper function to use Together's LLaMA-4 Maverick model
import together
import os

import os
from together import Together

# Initialize client using environment variable
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

def ask_llama(prompt, max_tokens=512):
    # Construct a conversation with a user prompt
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return "⚠️ Rate limit reached. Please wait 2 minutes before asking again."


# Detect file type
def get_file_type(filename):
    ext = filename.split('.')[-1].lower()
    if ext in ['csv', 'xlsx']: return "spreadsheet"
    elif ext in ['pdf']: return "pdf"
    elif ext in ['jpg', 'jpeg', 'png']: return "image"
    elif ext in ['txt']: return "text"
    elif ext in ['docx']: return "word"
    return "unknown"

# Extract content from uploaded file
def extract_text(file, file_type):
    if file_type == "spreadsheet":
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        return df
    elif file_type == "pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file_type == "image":
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        text = pytesseract.image_to_string(image)
        return text
    elif file_type == "text":
        return file.read().decode("utf-8")
    elif file_type == "word":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return "Unsupported format"

# Analyze DataFrame and show visualizations
def analyze_dataframe(df):
    st.subheader("📊 Basic Analysis")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())
    st.write("Summary:")
    st.dataframe(df.describe(include='all'))

    st.subheader("📌 Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📈 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# Ask questions to the model based on data
def answer_question_about_data(df, question):
    preview = df.head(10).to_csv(index=False)
    prompt = f"""
You are a data analyst.
You are a highly intelligent and helpful document analysis assistant.
The following document has been extracted using OCR/Text Parsing. Analyze the content and answer the question concisely.
Answer the data about the following the data can be text or an image or json
Here is a preview of the dataset:
{preview}

Answer the following question:
{question}
"""
    return ask_llama(prompt)


# Streamlit App UI
st.set_page_config(page_title=" DocuFlow AI: Data Analyst Agent", layout="wide")
st.title("DocuFlow AI: Data Analyst Agent")
st.markdown(
    "Upload any document (.csv, .xlsx, .pdf, .txt, .docx, image) to analyze data, generate visualizations, and ask questions."
)
# Sidebar navigation
page = st.sidebar.radio("Navigate to", ["Home", "About Us", "Contact Us"])

if page == "Home":
    st.header("Home")
    st.write("Welcome to the Multi-Agent File Classifier app. Upload your files and classify them!")
    uploaded_file = st.file_uploader(
        "📁 Upload your file", type=["csv", "xlsx", "pdf", "txt", "docx", "jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_type = get_file_type(uploaded_file.name)
        st.success(f"✅ Detected file type: `{file_type}`")

        content = extract_text(uploaded_file, file_type)

        # If file is a spreadsheet, show dataframe analysis and allow Q&A
        if file_type == "spreadsheet" and isinstance(content, pd.DataFrame):
            analyze_dataframe(content)

            st.subheader("🧠 Ask Questions About Your Data")
            question = st.text_input("❓ Enter your question:")
            if question:
                answer = answer_question_about_data(content, question)
                st.markdown(f"**Answer:** {answer}")

        # If file is a text-based type (pdf, docx, txt, image with OCR)
        elif isinstance(content, str):
            st.subheader("📄 Document Content Preview")
            st.text_area("Document Content", content[:3000], height=300)

            st.subheader("🧠 Ask Questions About This Document")
            question = st.text_input("❓ Enter your question:")
            if question:
                prompt = f"""You are a document analysis agent.
    Document:
    \"\"\"
    {content[:3000]}
    \"\"\"

    Question: {question}
    Answer:"""
                answer = ask_llama(prompt)
                st.markdown(f"**Answer:** {answer}")
        
        else:
            st.warning("⚠️ Unsupported file or failed to extract content.")
elif page == "About Us":
    st.header("About Us")

    st.markdown("""
    ### Welcome to **DocuFlow AI: Let Your Documents Flow Intelligently**

    At **DocuFlow AI**, we’re revolutionizing how businesses process and route their documents using the power of artificial intelligence. In a digital world overflowing with data, manual workflows just can’t keep up—and that’s where we step in.

    Our platform uses intelligent, collaborative agents to **analyze, classify, and act on documents** such as PDFs, JSONs, and Emails—automatically, securely, and efficiently.

    With just one upload, DocuFlow AI can detect the file type, extract key data, understand its intent (invoices, complaints, RFQs, etc.), analyze urgency and tone, and route it to the right place—all in real-time.

    --- 

    ### What We Do

    - 1 **Multi-format File Classification**: Supports PDFs, Emails, JSONs, and more.
    - 2 **Intent Detection**: Understands business context—complaint, invoice, payment, etc.
    - 3 **Urgency & Tone Analysis**: Prioritizes documents based on their emotional and urgency level.
    - 4 **Action Routing**: Escalates critical items to the right business systems (e.g., CRM).
    - 5 **Memory & Audit Logs**: Stores processing logs securely for compliance and performance tuning.

    ---

    ###  Our Vision

    We believe the future of work is **autonomous, smart, and fast**. Our mission is to eliminate repetitive manual tasks and unlock your team’s true potential by letting AI handle the heavy lifting.

    Whether you're a startup or an enterprise, DocuFlow AI helps you scale with confidence.

    ---

    ###  Why Choose DocuFlow AI?

    - ✨ **Built for Simplicity**: Clean UI with Streamlit for zero-friction use.
    - ✨ **Modular & Scalable**: Plug in your own agents and customize flows.
    - ✨ **Secure by Design**: Enterprise-ready and audit-compliant.
    - ✨ **Explainable AI**: We don’t do black boxes. Understand every decision made.

    ---

    **Transform the way you manage documents—join the flow with DocuFlow AI.**
    """)
elif page == "Contact Us":
    st.header("Contact Us")
    st.write("""
    Want to collaborate or try DocuFlow AI for your organization?

    - 🔗 GitHub Repository: [https://github.com/Nayan1801/DocuFlow-AI](https://github.com/Nayan1801/DocuFlow-AI)
    - 📧 Email: [nayankathait@gmail.com](mailto:nayankathait@gmail.com)
    - 🌐 Website (coming soon): [www.docuflow.ai](http://www.docuflow.ai) *(placeholder)*
    """)
