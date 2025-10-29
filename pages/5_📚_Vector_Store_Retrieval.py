import streamlit as st
import os
import tempfile
from pathlib import Path
import hashlib
import json
import re
from typing import List, Dict, Optional
import warnings
import time

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Document processing imports
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from fastembed import TextEmbedding

# Qdrant vector database
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import Distance, VectorParams

# Gemini API
import google.generativeai as genai

# Filter warnings
warnings.filterwarnings("ignore", message=".*'pin_memory' argument.*")

# Add debug logging toggle
DEBUG_MODE = True  # Set to True to enable debug logging, False to disable

def debug_log(message):
    """Helper function to log debug messages when DEBUG_MODE is True"""
    if DEBUG_MODE:
        print(f"[VECTOR_STORE_DEBUG] {message}")

# Initialize session state for document management
MANIFEST_FILE = "collection_manifest.json"

def load_document_manifest():
    """Load the document manifest from file if it exists"""
    debug_log("Loading document manifest...")
    start_time = time.time()
    
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, "r") as f:
                manifest = json.load(f)
                # Ensure the manifest is a dictionary
                if isinstance(manifest, dict):
                    debug_log(f"Loaded manifest with {len(manifest)} entries. Took {time.time() - start_time:.2f}s")
                    return manifest
                else:
                    st.warning("Document manifest file format is invalid, starting fresh.")
                    debug_log("Invalid manifest format, returning empty dict.")
                    return {}
        except Exception as e:
            st.error(f"Error loading document manifest: {str(e)}")
            debug_log(f"Error in load_document_manifest: {str(e)}")
            return {}
    
    debug_log("No manifest file found, returning empty dict.")
    return {}

def save_document_manifest(manifest):
    """Save the document manifest to file"""
    debug_log("Saving document manifest...")
    start_time = time.time()
    
    try:
        with open(MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=4)
        
        debug_log(f"Saved manifest with {len(manifest)} entries. Took {time.time() - start_time:.2f}s")
    except Exception as e:
        st.error(f"Error saving document manifest: {str(e)}")
        debug_log(f"Error in save_document_manifest: {str(e)}")

debug_log("Starting page initialization...")
debug_log("Loading manifest...")
# Load the manifest on app start
if 'document_manifest' not in st.session_state:
    st.session_state.document_manifest = load_document_manifest()
else:
    # Even if already in session state, reload from file to ensure consistency after restarts
    fresh_manifest = load_document_manifest()
    # Update session state manifest with fresh data, but keep any new entries during this session
    st.session_state.document_manifest.update(fresh_manifest)

# Ensure the manifest is properly displayed after loading
if st.session_state.document_manifest:
    debug_log(f"Manifest has {len(st.session_state.document_manifest)} documents.")
    # st.rerun() # Commented out to avoid auto-rerun

if 'qdrant_client' not in st.session_state:
    st.session_state.qdrant_client = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

st.set_page_config(
    page_title="Vector Store Retrieval",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Vector Store Retrieval System")
st.markdown("""
This system allows you to upload documents, index them using vector embeddings, and then ask questions about the content.
The AI will only respond based on the information in your documents.
""")

# Initialize Qdrant client
@st.cache_resource
def get_qdrant_client():
    """Initialize and return Qdrant client"""
    debug_log("Initializing Qdrant client...")
    start_time = time.time()
    
    try:
        # Check if another instance is using the storage and provide a more graceful error
        client = QdrantClient(path="./qdrant_data")  # Local persistent storage
        elapsed = time.time() - start_time
        debug_log(f"Qdrant client initialized successfully. Took {elapsed:.2f}s")
        
        if DEBUG_MODE:
            st.success("Qdrant client initialized successfully!")
        return client
    except Exception as e:
        error_msg = f"Error initializing Qdrant client: {str(e)}"
        st.error(error_msg)
        
        # Check if the error is related to concurrent access
        if "already accessed by another instance" in str(e):
            st.warning("Qdrant is being used by another process. Please close other Streamlit instances or use Qdrant server for concurrent access.")
        
        debug_log(f"Error in get_qdrant_client: {str(e)}")
        return None

# Initialize embedding model
@st.cache_resource
def get_embedding_model():
    """Initialize and return the embedding model"""
    debug_log("Initializing embedding model...")
    start_time = time.time()
    
    try:
        # Using a lightweight but effective embedding model
        # This will be downloaded only once and cached
        model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir="./.cache")
        elapsed = time.time() - start_time
        debug_log(f"Embedding model loaded successfully. Took {elapsed:.2f}s")
        
        if DEBUG_MODE:
            st.success("Embedding model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        debug_log(f"Error in get_embedding_model: {str(e)}")
        return None

# Initialize chunker
@st.cache_resource
def get_chunker():
    """Initialize and return the chunker"""
    debug_log("Initializing chunker...")
    start_time = time.time()
    
    try:
        chunker = HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2")
        elapsed = time.time() - start_time
        debug_log(f"Chunker loaded successfully. Took {elapsed:.2f}s")
        
        if DEBUG_MODE:
            st.success("Chunker loaded successfully!")
        return chunker
    except Exception as e:
        st.error(f"Error loading chunker: {str(e)}")
        debug_log(f"Error in get_chunker: {str(e)}")
        return None

# Document processing system
class DocumentProcessor:
    def __init__(self, qdrant_client: QdrantClient, embedding_model: TextEmbedding, chunker: HybridChunker):
        debug_log("Initializing DocumentProcessor...")
        self.client = qdrant_client
        self.embedding_model = embedding_model
        self.chunker = chunker
        self.converter = DocumentConverter()
        debug_log("DocumentProcessor initialized.")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Creates a valid Qdrant collection name from a filename."""
        # Remove extension name = os.path.splitext(filename)[0]
        name = os.path.splitext(filename)[0]
        # Replace non-alphanumeric characters with underscores
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it doesn't start/end with underscores
        return name.strip('_').lower()
    
    def process_document(self, file_path: str, original_filename: str) -> bool:
        """Process a document by converting, chunking, and indexing it."""
        debug_log(f"Processing document: {original_filename}")
        start_time = time.time()
        
        collection_name = self._sanitize_filename(original_filename)
        
        try:
            file_ext = Path(file_path).suffix.lower()
            debug_log(f"Processing file extension: {file_ext}")
            
            # Handle different file types appropriately
            if file_ext == '.txt':
                # For text files, read directly
                debug_log("Processing as text file...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                # Split text into chunks using regex
                # This is a simple approach that respects sentence boundaries
                paragraphs = [p.strip() for p in text_content.split('\n') if p.strip()]
                text_chunks = []
                
                # Combine paragraphs to make chunks of appropriate size
                current_chunk = ""
                for paragraph in paragraphs:
                    if len(current_chunk) + len(paragraph) < 1000:  # Max chunk size
                        current_chunk += paragraph + "\n\n"
                    else:
                        if current_chunk:
                            text_chunks.append(current_chunk.strip())
                            current_chunk = paragraph + "\n\n"
                
                if current_chunk.strip():
                    text_chunks.append(current_chunk.strip())
            else:
                # For other supported formats (PDF, DOCX), use Docling
                debug_log("Processing as non-text file with Docling...")
                document = self.converter.convert(source=file_path).document
                
                # Chunk the document
                chunks = self.chunker.chunk(dl_doc=document, max_tokens=300)
                text_chunks = [c.text for c in chunks]
            
            debug_log(f"Got {len(text_chunks)} text chunks.")
            
            if not text_chunks:
                st.error("Document could not be chunked or contains no text.")
                debug_log("Document has no chunks, returning False.")
                return False
            
            # Create embeddings
            debug_log("Creating embeddings...")
            embedding_start = time.time()
            embeddings = list(self.embedding_model.embed(text_chunks))
            embeddings_dim = len(embeddings[0])
            debug_log(f"Embeddings created in {time.time() - embedding_start:.2f}s, dim: {embeddings_dim}")
            
            # Create collection
            debug_log("Creating Qdrant collection...")
            collection_start = time.time()
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embeddings_dim, distance=Distance.COSINE)
            )
            debug_log(f"Collection created in {time.time() - collection_start:.2f}s")
            
            # Upload points to Qdrant
            debug_log("Uploading points to Qdrant...")
            upload_start = time.time()
            points = [
                models.PointStruct(id=i, vector=vec, payload={"text": text_chunks[i]})
                for i, vec in enumerate(embeddings)
            ]
            self.client.upload_points(collection_name=collection_name, points=points)
            debug_log(f"Points uploaded in {time.time() - upload_start:.2f}s")
            
            # Update session state manifest
            debug_log("Updating manifest...")
            st.session_state.document_manifest[original_filename] = {
                "collection_name": collection_name,
                "chunks_count": len(text_chunks)
            }
            
            # Save manifest to file for persistence across sessions
            save_document_manifest(st.session_state.document_manifest)
            debug_log(f"Document processed in total {time.time() - start_time:.2f}s")
            
            return True
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            debug_log(f"Error in process_document: {str(e)}")
            return False

# RAG (Retrieval-Augmented Generation) system
class RAGSystem:
    def __init__(self, qdrant_client: QdrantClient, embedding_model: TextEmbedding):
        debug_log("Initializing RAGSystem...")
        start_time = time.time()
        
        self.client = qdrant_client
        self.embedding_model = embedding_model
        
        # Configure Generative AI from .env
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            st.error("GEMINI_API_KEY not found in .env file. Please add it to the .env file.")
        genai.configure(api_key=gemini_api_key)
        
        # Get the model from .env, default to gemini-flash-latest if not specified
        gemini_model_name = os.getenv("MODEL", "gemini-flash-latest")
        self.model = genai.GenerativeModel(gemini_model_name)
        
        debug_log(f"RAGSystem initialized in {time.time() - start_time:.2f}s")
    
    def retrieve_context(self, query: str, collection_name: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant chunks from the vector database."""
        debug_log(f"Retrieving context for query: '{query[:50]}...' from collection: {collection_name}")
        start_time = time.time()
        
        try:
            query_embedding_start = time.time()
            query_vector = list(self.embedding_model.embed([query]))[0]
            debug_log(f"Query embedding created in {time.time() - query_embedding_start:.2f}s")
            
            # Search in Qdrant
            search_start = time.time()
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
            )
            debug_log(f"Search completed in {time.time() - search_start:.2f}s")
            
            # Extract text from search results
            context = [hit.payload["text"] for hit in search_results]
            debug_log(f"Retrieved {len(context)} context chunks in {time.time() - start_time:.2f}s")
            return context
        except Exception as e:
            st.error(f"Error retrieving context: {str(e)}")
            debug_log(f"Error in retrieve_context: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context: List[str], doc_name: str) -> str:
        """Generate an answer based on the query and retrieved context."""
        debug_log(f"Generating answer for query: '{query[:50]}...' with {len(context)} context chunks")
        start_time = time.time()
        
        if not context:
            debug_log("No context found, returning default message")
            return "I couldn't find any relevant information in the document to answer your question."
        
        # Format the context
        context_text = "\n\n".join(context)
        
        # Create the prompt for Gemini
        prompt = f"""
        You are a helpful assistant that answers questions based only on the provided context.
        If the answer is not in the context, clearly state that you don't know the answer based on the provided information.
        
        Context from document '{doc_name}':
        {context_text}
        
        Question: {query}
        
        Answer:
        """
        
        try:
            response = self.model.generate_content(prompt)
            debug_log(f"Answer generated in {time.time() - start_time:.2f}s")
            return response.text
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            debug_log(f"Error in generate_answer: {str(e)}")
            return "Sorry, I encountered an error while generating the answer."

with st.spinner("Initializing components... This may take a moment on first run as models are downloaded."):
    # Initialize components
    debug_log("Starting component initialization...")
    init_start_time = time.time()
    
    qdrant_client = get_qdrant_client()
    embedding_model = get_embedding_model()
    chunker = get_chunker()

    if qdrant_client is None or embedding_model is None or chunker is None:
        st.error("Failed to initialize required components. Please check your configuration.")
        debug_log("Failed to initialize components, stopping execution.")
        st.stop()

    # Initialize DocumentProcessor and RAGSystem
    doc_processor = DocumentProcessor(qdrant_client, embedding_model, chunker)

    # Check for API key in .env
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.warning("GEMINI_API_KEY not found in .env file. Please add it to the .env file to use the Q&A functionality.")
        debug_log("No GEMINI_API_KEY found, stopping execution.")
        st.stop()

    rag_system = RAGSystem(qdrant_client, embedding_model)
    
    debug_log(f"All components initialized in {time.time() - init_start_time:.2f}s")

debug_log("Component initialization complete.")

# File uploader for documents
st.header("Upload Documents")
uploaded_files = st.file_uploader(
    "Choose documents to index (PDF, TXT, DOCX)",
    type=['pdf', 'txt', 'docx'],
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files:
    debug_log(f"Processing {len(uploaded_files)} uploaded files...")
    processed_files = []
    for uploaded_file in uploaded_files:
        # Check if document is already processed
        if uploaded_file.name in st.session_state.document_manifest:
            st.info(f"Document '{uploaded_file.name}' already processed. Skipping...")
            continue
        
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Process the document using Docling
                if doc_processor.process_document(temp_path, uploaded_file.name):
                    st.success(f"Successfully processed '{uploaded_file.name}'")
                    processed_files.append(uploaded_file.name)
                else:
                    st.error(f"Failed to process '{uploaded_file.name}'")
                
                # Remove temporary file
                os.unlink(temp_path)
            except Exception as e:
                st.error(f"Error processing '{uploaded_file.name}': {str(e)}")
    
    if processed_files:
        st.success(f"Processed {len(processed_files)} documents successfully!")

# Display indexed documents
st.header("Indexed Documents")
if st.session_state.document_manifest:
    debug_log(f"Displaying {len(st.session_state.document_manifest)} indexed documents...")
    cols = st.columns(3)
    for i, (filename, info) in enumerate(st.session_state.document_manifest.items()):
        col = cols[i % 3]
        with col:
            st.info(f"📄 {filename}")
            st.caption(f"Chunks: {info['chunks_count']}")
            
            # Add delete button for each document
            if st.button(f"Delete", key=f"delete_{filename}"):
                try:
                    # Remove collection from Qdrant
                    qdrant_client.delete_collection(info['collection_name'])
                    # Remove from manifest
                    del st.session_state.document_manifest[filename]
                    # Save updated manifest
                    save_document_manifest(st.session_state.document_manifest)
                    st.success(f"Deleted '{filename}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting '{filename}': {str(e)}")
else:
    st.info("No documents indexed yet. Upload documents using the section above.")

# Query interface
st.header("Ask Questions")
if st.session_state.document_manifest:
    # Document selection
    doc_options = list(st.session_state.document_manifest.keys())
    selected_filename = st.selectbox("Select a document to query:", doc_options)
    selected_collection = st.session_state.document_manifest[selected_filename]["collection_name"]
    
    # Query input
    query = st.text_input("Enter your question:")
    
    if query:
        debug_log(f"Processing query: '{query[:50]}...' for collection {selected_collection}")
        with st.spinner("Searching and generating answer..."):
            # Retrieve context from vector database
            context = rag_system.retrieve_context(query, selected_collection, top_k=5)
            
            if context:
                # Generate answer using RAG system
                answer = rag_system.generate_answer(query, context, selected_filename)
                
                # Display answer
                st.subheader("Answer:")
                st.write(answer)
                
                # Display context used
                st.subheader("Context Used:")
                for i, context_text in enumerate(context):
                    with st.expander(f"Context {i+1}"):
                        st.write(context_text)
            else:
                st.warning("No relevant information found in the selected document.")
else:
    st.info("Please upload and index documents first before asking questions.")
    
debug_log("Page rendering complete.")