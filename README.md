# Multi-Project Streamlit AI Application

This repository contains a collection of AI-powered mini-applications built with Streamlit. The main application serves as a launchpad for various projects, each demonstrating different capabilities of Large Language Models (LLMs) and related technologies. You can seamlessly switch between projects using the sidebar navigation.

## ✨ Features

Currently, the following projects are available:

### 🤖 LLM Chatbot
A versatile chat interface that allows you to interact with different LLMs.
- **Provider Agnostic:** Connect to local models via [Ollama](https://ollama.com/) or to cloud-based models like Google's [Gemini](https://ai.google.dev/).
- **Document Q&A:** Upload a document (PDF, TXT, or MD) to chat with its content. The model will use the document as the primary context to answer your questions, ensuring responses are grounded in the provided text.
- **Real-time Interaction:** A clean, intuitive chat interface for seamless conversations.

### 📄 Document Summarizer
An efficient tool to generate detailed, structured summaries of your documents.
- **Multiple File Types:** Supports PDF, TXT, and Markdown files.
- **Choice of Model:** Leverage either a local Ollama model or the Gemini API to generate summaries.
- **Structured Output:** The summary is presented with clear headings and bullet points for enhanced readability.
- **Streaming Responses:** Watch the summary get generated in real-time for a better user experience.

### 📰 News Summarizer
A real-time news summarizer that fetches trending news articles and generates AI summaries using the GNews API.
- **GNews API Integration:** Uses the GNews API with your personal API key (configured in .env file) to fetch news
- **Category & Search:** Browse by categories (General, World, Business, Technology, etc.) or search for specific topics
- **Language & Country Filters:** Select news in different languages and from specific countries
- **Date Filtering:** Filter news by date range (Today, This week, This month, or All time)
- **AI Summaries:** Generate summaries using either Gemini or Ollama models
- **Database Caching:** News articles and AI summaries are cached in a local SQLite database to reduce API calls and improve performance
- **Rate Limiting Handling:** Implements retry logic with exponential backoff to handle API rate limits gracefully
- **Image Support:** Displays article images with proper container width

### 🎙️ Meeting Notes & Action Item Extractor
Converts meeting audio recordings into structured notes and action items.
- **Audio Support:** Upload audio files in MP3, WAV, M4A, MP4, M4V, MOV formats
- **Whisper Transcription:** Uses local Whisper model for speech-to-text conversion with chunked processing for better performance
- **AI Processing:** Generates structured meeting summaries with key points, action items, decisions, and next steps
- **Chunked Processing:** Processes long audio files in smaller chunks for better progress tracking and memory management
- **Language Support:** Offers language selection for improved transcription accuracy
- **Structured Output:** Provides organized sections for meeting points, action items, decisions, and next steps

### 📚 Vector Store Retrieval
An RAG (Retrieval-Augmented Generation) system that enables semantic search and Q&A with your documents using vector embeddings.
- **Similarity Search:** Uses vector embeddings to find semantically relevant document chunks
- **Multi-Model Support:** Supports both local Ollama and cloud-based Gemini models
- **Qdrant Integration:** Utilizes Qdrant as a vector database for efficient storage and retrieval
- **Document Ingestion:** Indexes documents (PDF, TXT, MD) into the vector store for future retrieval
- **Contextual Responses:** Generates responses based on retrieved document content with source citations

## 🛠️ Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Prerequisites
- Python 3.8 or higher.
- (Optional but Recommended) [Ollama](https://ollama.com/) installed and running locally if you wish to use local models. You can pull a model by running `ollama pull llama3`.
- [Qdrant](https://qdrant.tech/) (for Vector Store Retrieval features) - can be run locally or via Docker

### 2. Clone the Repository
```bash
git clone https://github.com/Mathew005/Multi-Project-Streamlit-AI-Application.git
cd Multi-Project-Streamlit-AI-Application
```

### 3. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install the packages from the requirements.txt file:
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create a file named `.env` in the root directory of the project and populate it with your API keys and configuration.

```dotenv
# .env file

# --- Ollama Configuration ---
# This should point to your local Ollama instance
OLLAMA_BASE_URL=http://127.0.0.1:11434

# --- Gemini Configuration ---
# Get your API key from https://aistudio.google.com/app/apikey
# Make sure to paste your actual key here
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

# --- Global Model Configuration ---
# Model to use for all AI operations
MODEL="gemini-flash-latest"

# --- GNews API Configuration ---
# Get your API key from https://gnews.io/
GNEWS_API_KEY=your-gnews-api-key-here
```
**Important:** Replace `"YOUR_GEMINI_API_KEY_HERE"` with your actual Gemini API key and update other API keys as needed.

## ▶️ How to Run the Application

With your environment set up and the `.env` file configured, start the Streamlit application with a single command:

```bash
streamlit run main.py
```

A new tab should open in your web browser at `http://localhost:8501`.

- **To navigate between projects**, use the sidebar on the left.
- **To configure a project**, use the options available in its respective sidebar.

## 📂 Project Structure

The project is organized to support multiple pages, following Streamlit's multi-page app structure.

```
.
├── .env                          # Environment variables for API keys and configs
├── collection_manifest.json      # Qdrant collection manifest
├── main.py                       # The main entry point for the Streamlit app (Welcome Page)
├── pages/                        # Directory for individual Streamlit project pages
│   ├── 1_🤖_LLM_Chatbot.py
│   ├── 2_📄_Document_Summarizer.py
│   ├── 3_🎙️_Meeting_Notes_Extractor.py
│   ├── 4_📰_News_Summarizer.py
│   └── 5_📚_Vector_Store_Retrieval.py
├── qdrant_data/                  # Local Qdrant vector database storage
├── requirements.txt              # List of Python dependencies
├── .cache/                       # Model cache directory
├── .gitignore
└── README.md                     # This file
```

## Images

<img width="1788" height="848" alt="image" src="https://github.com/user-attachments/assets/e6280dad-f28b-45dd-8822-a677db180150" />

<img width="1788" height="856" alt="image" src="https://github.com/user-attachments/assets/82ec7f81-eb24-45f5-a967-c90acee2399f" />
