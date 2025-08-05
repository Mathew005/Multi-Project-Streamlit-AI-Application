import streamlit as st
import requests
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO

# --- Page and Environment Configuration ---
load_dotenv()
st.set_page_config(page_title="LLM Chat", page_icon="🤖", layout="wide")
st.title("🤖 LLM Chatbot")
st.caption("Chat with a model or upload a document for contextual Q&A.")


# --- Helper Function to Extract Text ---
def extract_text_from_file(uploaded_file):
    if not uploaded_file: return None
    try:
        bytes_data = uploaded_file.getvalue()
        if uploaded_file.name.lower().endswith('.pdf'):
            reader = PyPDF2.PdfReader(BytesIO(bytes_data))
            return "".join(page.extract_text() or "" for page in reader.pages)
        elif uploaded_file.name.lower().endswith(('.txt', '.md')):
            return bytes_data.decode('utf-8')
    except Exception as e:
        st.error(f"Error reading or processing file: {e}")
    return None

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
if "document_context" not in st.session_state:
    st.session_state.document_context = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    model_provider = st.selectbox("Choose a provider:", ["Gemini", "Ollama"])
    
    # Model Selection UI
    selected_model = None
    if model_provider == "Ollama":
        try:
            res = requests.get("http://localhost:11434/api/tags")
            res.raise_for_status()
            models = [m["name"] for m in res.json().get("models", [])]
            if models:
                selected_model = st.selectbox("Choose an Ollama model:", models)
            else:
                st.warning("No Ollama models found.")
        except requests.exceptions.RequestException:
            st.error("Could not connect to Ollama.")
    
    elif model_provider == "Gemini":
        if os.getenv("GEMINI_API_KEY"):
            st.success("Gemini API Key loaded.")
            selected_model = "gemini/gemini-1.5-flash-latest"
        else:
            st.error("GEMINI_API_KEY not found.")

    st.divider()

    # File Uploader and Context Management
    st.header("Document Context")
    uploaded_file = st.file_uploader(
        "Upload a file for contextual chat",
        type=["pdf", "txt", "md"],
        key="file_uploader_key" # Unique key
    )

    if st.button("Load Document"):
        if uploaded_file:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                extracted_text = extract_text_from_file(uploaded_file)
                if extracted_text:
                    st.session_state.document_context = extracted_text
                    st.session_state.document_name = uploaded_file.name
                    st.success(f"✅ Context from '{uploaded_file.name}' loaded.")
                else:
                    st.error("Could not extract text from the document.")
        else:
            st.warning("Please upload a file first.")

    if st.session_state.document_name:
        st.info(f"Active Document: **{st.session_state.document_name}**")
        if st.button("Clear Document Context"):
            st.session_state.document_context = None
            st.session_state.document_name = None
            st.rerun()


# --- Main Chat Interface ---
# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle new chat input
if prompt := st.chat_input("Ask a question..."):
    if not selected_model:
        st.info("Please select a model from the sidebar.")
        st.stop()

    # Add user's message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # **This is the critical logic inspired by your backend.py**
    prompt_for_llm = prompt
    if st.session_state.document_context:
        prompt_for_llm = (
            f"You are an expert assistant. A user has provided you with a document to analyze. "
            f"Answer the user's question based *only* on the context provided in the document.\n\n"
            f"--- DOCUMENT CONTEXT ---\n"
            f"{st.session_state.document_context}\n"
            f"--- END OF CONTEXT ---\n\n"
            f"User's Question: {prompt}"
        )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text = None
            try:
                # --- OLLAMA API CALL ---
                if model_provider == "Ollama":
                    payload = {"model": selected_model, "messages": [{"role": "user", "content": prompt_for_llm}], "stream": False}
                    res = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
                    res.raise_for_status()
                    response_text = res.json()["message"]["content"]

                # --- GEMINI API CALL ---
                elif model_provider == "Gemini":
                    api_key = os.getenv("GEMINI_API_KEY")
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash-latest')
                    response = model.generate_content(prompt_for_llm)
                    response_text = response.text
                
                if response_text:
                    st.write(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    st.error("The model did not return a response.")

            except requests.exceptions.RequestException as e:
                st.error(f"API Request Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")