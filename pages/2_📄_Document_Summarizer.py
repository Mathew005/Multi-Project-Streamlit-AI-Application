import streamlit as st
import requests
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO

# --- Page and Environment Configuration ---
load_dotenv()
st.set_page_config(page_title="Document Summarizer", page_icon="📄", layout="wide")
st.title("📄 Document Summarizer")
st.caption("Upload a document and get a detailed summary using your chosen LLM.")


# --- Helper Function to Extract Text ---
def extract_text_from_file(uploaded_file):
    """Extracts text from an uploaded file (.pdf, .txt, .md)."""
    if not uploaded_file:
        return None
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


# --- Sidebar for Model and Provider Selection ---
with st.sidebar:
    st.header("Configuration")
    model_provider = st.selectbox("Choose a provider:", ["Ollama", "Gemini"])
    
    selected_model = None
    # --- Ollama Model Selection ---
    if model_provider == "Ollama":
        try:
            res = requests.get("http://localhost:11434/api/tags")
            res.raise_for_status()
            models = [m["name"] for m in res.json().get("models", [])]
            if models:
                selected_model = st.selectbox("Choose an Ollama model:", models)
            else:
                st.warning("No Ollama models found. Please run `ollama pull <model>`.")
        except requests.exceptions.RequestException:
            st.error("Could not connect to Ollama. Please ensure it is running.")
    
    # --- Gemini Model Selection ---
    elif model_provider == "Gemini":
        if os.getenv("GEMINI_API_KEY"):
            st.success("Gemini API Key loaded successfully.")
            # Use the model specified in the .env file, with proper prefixing
            model_name = os.getenv("MODEL", "gemini-flash-latest")  # Default fallback if MODEL is not set
            selected_model = f"gemini/{model_name}" if not model_name.startswith("gemini/") else model_name
            st.info(f"Using model: **{selected_model.split('/')[-1]}**")
        else:
            st.error("GEMINI_API_KEY not found in your .env file.")


# --- Main Page UI ---
uploaded_file = st.file_uploader(
    "Upload your document (.pdf, .txt, .md)",
    type=["pdf", "txt", "md"],
    key="summarizer_file_uploader"
)


if st.button("Summarize Document", type="primary"):
    # 1. Validate inputs
    if not uploaded_file:
        st.warning("Please upload a document first.")
    elif not selected_model:
        st.warning("Please select a valid model in the sidebar.")
    else:
        # 2. Process the document and prepare for summarization
        with st.spinner(f"Reading '{uploaded_file.name}'..."):
            extracted_text = extract_text_from_file(uploaded_file)
        
        if not extracted_text or not extracted_text.strip():
            st.error("Could not extract any text from the document. The file might be empty or corrupted.")
        else:
            # 3. Create the detailed prompt
            summarization_prompt = f"""
            As an expert summarizer, create a detailed, structured summary of the following document. The summary should capture all main arguments, key evidence, and conclusions, using clear headings and bullet points for readability. Maintain a neutral tone and be faithful to the original text.

            Document Content:
            ---
            {extracted_text}
            ---

            Provide the detailed summary:
            """
            
            st.subheader("📄 Summary")
            summary_placeholder = st.empty()
            
            try:
                # --- OLLAMA API CALL (with streaming) ---
                if model_provider == "Ollama":
                    st.info("Generating summary with Ollama... (Response will stream in)")
                    payload = {
                        "model": selected_model,
                        "messages": [{"role": "user", "content": summarization_prompt}],
                        "stream": True # Enable streaming
                    }
                    full_response = ""
                    with requests.post("http://localhost:11434/api/chat", json=payload, stream=True, timeout=300) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if line:
                                try:
                                    # Each line is a JSON object, parse it
                                    chunk = json.loads(line)
                                    # Append the content from the 'message' part of the chunk
                                    content_piece = chunk["message"]["content"]
                                    full_response += content_piece
                                    # Update the placeholder with the latest full response
                                    summary_placeholder.markdown(full_response + "▌")
                                except json.JSONDecodeError:
                                    st.error(f"Failed to decode a line from the stream: {line}")
                                    break
                    summary_placeholder.markdown(full_response)


                # --- GEMINI API CALL (non-streaming, as it's generally fast) ---
                elif model_provider == "Gemini":
                    with st.spinner("Generating summary with Gemini..."):
                        api_key = os.getenv("GEMINI_API_KEY")
                        genai.configure(api_key=api_key)
                        model_name = os.getenv("MODEL", "gemini-flash-latest")  # Default fallback if MODEL is not set
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(summarization_prompt, stream=True)
                        
                        full_response = ""
                        for chunk in response:
                             full_response += chunk.text
                             summary_placeholder.markdown(full_response + "▌")
                        summary_placeholder.markdown(full_response)
                
                if not full_response:
                    st.error("The model did not return a summary.")

            except requests.exceptions.RequestException as e:
                st.error(f"API Request Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")