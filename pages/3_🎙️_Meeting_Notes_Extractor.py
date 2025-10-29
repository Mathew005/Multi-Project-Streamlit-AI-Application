import streamlit as st
import requests
import os
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
import wave
from pydub import AudioSegment
import io
import subprocess
import sys
from typing import Optional
import time
import math

# --- Page and Environment Configuration ---
load_dotenv()
st.set_page_config(page_title="Meeting Notes Extractor", page_icon="🎙️", layout="wide")
st.title("🎙️ Meeting Notes & Action Item Extractor")
st.caption("Upload meeting audio and get structured notes with action items using Whisper and Gemini API.")

# --- Introduction ---
st.markdown("""
This tool converts your meeting audio recordings into structured notes and action items. 

**How it works:**
1. Upload your meeting audio file (MP3, WAV, M4A, MP4, M4V, MOV)
2. The audio is transcribed using Whisper speech recognition
3. Gemini AI processes the transcript to extract key points, action items, and decisions
4. View and download the structured meeting summary

*Note: Processing may take several minutes depending on audio length.*
""")


# --- Whisper Transcription Function ---
def transcribe_audio_with_whisper(audio_file_path: str, progress_bar) -> str:
    """
    Transcribe audio using faster-whisper library with chunked processing for better performance
    """
    try:
        from faster_whisper import WhisperModel
        import tempfile
        from pydub import AudioSegment
        
        # Determine the appropriate model based on available hardware
        model_size = st.session_state.get("whisper_model", "base")
        
        # Show model loading progress
        with st.spinner(f"Loading Whisper model ({model_size})... This may take a moment."):
            # Initialize the Whisper model
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        # Load the audio file
        audio = AudioSegment.from_file(audio_file_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Set to 16kHz which is optimal for Whisper
        audio = audio.set_frame_rate(16000)
        
        # Calculate duration and determine chunk size (in milliseconds)
        duration_ms = len(audio)
        chunk_duration_ms = 60000  # 1 minute per chunk (in milliseconds) - smaller chunks for better progress tracking
        
        st.info(f"Audio duration: {duration_ms/60000:.1f} minutes. Processing in chunks of up to {chunk_duration_ms/60000:.0f} minute(s) each...")
        
        transcription = ""
        
        # Process in smaller chunks sequentially to allow more frequent progress updates
        # Whisper is CPU-intensive, so we limit to 1 concurrent task to avoid overwhelming the system
        total_chunks = math.ceil(duration_ms / chunk_duration_ms)
        for i, start_ms in enumerate(range(0, duration_ms, chunk_duration_ms), 1):
            end_ms = min(start_ms + chunk_duration_ms, duration_ms)
            
            # Get the current chunk
            chunk = audio[start_ms:end_ms]
            
            # Save chunk temporarily to a file for Whisper processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
                chunk.export(chunk_file.name, format="wav")
                
                # Transcribe the chunk with selected language (None allows auto-detection)
                segments, info = model.transcribe(
                    chunk_file.name, 
                    beam_size=5, 
                    language=st.session_state.get("audio_language", None)  # Use selected language, None allows auto-detection
                )
                
                # Clean up temp file
                os.unlink(chunk_file.name)
            
            # Collect the transcribed text for this chunk
            chunk_transcription = ""
            for segment in segments:
                chunk_transcription += segment.text + " "
            
            transcription += chunk_transcription + " "
            
            # Calculate progress
            progress = min(end_ms / duration_ms, 1.0)
            progress_bar.progress(progress, text=f"Processing chunk {i} of {total_chunks}")
        
        return transcription.strip()
    except ImportError:
        st.error("faster-whisper library is not installed. Please install it with: `pip install faster-whisper`")
        st.info("On some systems you might need to install additional dependencies. Try: `pip install faster-whisper[cuda]` if you have a compatible GPU.")
        return ""
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        st.info("Note: Larger audio files may take several minutes to process. If the error persists, try a smaller file or a different Whisper model size.")
        return ""


# --- AI Processing Function ---
def extract_meeting_notes(transcript: str, provider: str, model_name: str) -> Optional[dict]:
    """
    Extract structured meeting notes and action items using AI
    """
    try:
        if provider == "Gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("GEMINI_API_KEY not found in environment variables")
                return None
            
            genai.configure(api_key=api_key)
            # Use the model specified in the .env file
            model_name = os.getenv("MODEL", "gemini-flash-latest")  # Default fallback if MODEL is not set
            model = genai.GenerativeModel(model_name)
            
            prompt = f"""
            Analyze the following meeting transcript and provide a structured summary. 
            Format your response clearly with these sections:
            
            ## Key Meeting Points
            - List the main topics discussed
            
            ## Action Items
            - Task with assigned person and deadline if mentioned
            
            ## Decisions Made
            - List all decisions made during the meeting
            
            ## Next Steps
            - Future actions or meetings scheduled
            
            Transcript:
            {transcript[:4000]}  # Limit transcript length to avoid exceeding context window
            """
            
            response = model.generate_content(prompt)
            
            return {"raw_response": response.text}
            
        elif provider == "Ollama":
            payload = {
                # Use the model_name directly without any prefix for Ollama
                "model": model_name,
                "prompt": f"""
                Analyze the following meeting transcript and provide a structured summary.
                
                Transcript: {transcript[:4000]}  # Limit transcript length
                
                Format your response with:
                - Key Meeting Points (list main topics discussed)
                - Action Items (with assignee and deadline if mentioned)
                - Decisions Made (list all decisions made during the meeting)
                - Next Steps (future actions or meetings scheduled)
                """,
                "stream": False
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300)
            response.raise_for_status()
            
            return {"raw_response": response.json()["response"]}
        
        return None
    except Exception as e:
        st.error(f"Error during AI processing: {str(e)}")
        return None


# --- Session State Initialization ---
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "meeting_notes" not in st.session_state:
    st.session_state.meeting_notes = None


# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    model_provider = st.selectbox("Choose an AI provider:", ["Gemini", "Ollama"])
    
    # Model Selection UI
    selected_model = None
    if model_provider == "Ollama":
        try:
            res = requests.get("http://localhost:11434/api/tags")
            res.raise_for_status()
            models = [m["name"] for m in res.json().get("models", [])]
            if models:
                selected_model = st.selectbox("Choose an Ollama model:", models, index=0)
            else:
                st.warning("No Ollama models found.")
        except requests.exceptions.RequestException:
            st.error("Could not connect to Ollama.")
    
    elif model_provider == "Gemini":
        if os.getenv("GEMINI_API_KEY"):
            st.success("Gemini API Key loaded.")
            # Use the model specified in the .env file
            selected_model = os.getenv("MODEL", "gemini-flash-latest")  # Default fallback if MODEL is not set
        else:
            st.error("GEMINI_API_KEY not found.")

    st.divider()
    
    # Whisper Model Selection
    st.header("Whisper Configuration")
    whisper_model = st.selectbox(
        "Choose Whisper model size:",
        ["tiny", "base", "small", "medium", "large-v1", "large-v2"],
        index=1
    )
    st.session_state.whisper_model = whisper_model
    
    # Language Selection
    language_options = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese": "zh",
        "Japanese": "ja",
        "Korean": "ko",
        "Russian": "ru",
        "Portuguese": "pt",
        "Italian": "it",
        "Dutch": "nl",
        "Arabic": "ar",
        "Hindi": "hi"
    }
    
    selected_language = st.selectbox(
        "Audio Language (for better accuracy):",
        options=list(language_options.keys()),
        index=0  # Default to English
    )
    
    # Store the selected language code
    st.session_state.audio_language = language_options[selected_language]
    
    st.info(f"Selected Whisper model: **{whisper_model}**")
    st.info(f"Selected language: **{selected_language}**")
    st.caption("Larger models are more accurate but slower")


# --- Main Interface ---
st.header("Upload Meeting Audio")

uploaded_file = st.file_uploader(
    "Choose an audio file", 
    type=["mp3", "wav", "m4a", "mp4", "m4v", "mov"],
    help="Supported formats: MP3, WAV, M4A, MP4, M4V, MOV"
)

if uploaded_file is not None:
    # Validate file format
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    supported_formats = [".mp3", ".wav", ".m4a", ".mp4", ".m4v", ".mov"]
    
    if file_extension not in supported_formats:
        st.error(f"Unsupported file format: {file_extension}. Please upload a file with one of these formats: {', '.join(supported_formats)}")
        st.stop()
    
    # Display audio player
    st.audio(uploaded_file, format="audio/wav")
    
    # Get file size and warn if too large
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    st.info(f"File size: {file_size_mb:.2f} MB")
    
    if file_size_mb > 50:  # More than 50MB
        st.warning("Large audio files may take a long time to process. Consider trimming the audio if possible.")
    elif file_size_mb > 100:
        st.error("Very large audio files may cause processing issues. Consider splitting the audio into smaller segments.")
    
    # Process audio when user clicks the button
    if st.button("Process Meeting Audio", type="primary"):
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Check if Whisper model is available
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            st.error("faster-whisper library is not installed. Please install it with: pip install faster-whisper")
            st.stop()
        
        # Save uploaded file temporarily
        status_text.text("Saving uploaded file...")
        progress_bar.progress(5)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_filename = tmp_file.name
        
        # Convert audio to WAV format if needed (Whisper works best with WAV)
        status_text.text("Converting audio to WAV format...")
        progress_bar.progress(10)
        
        # Convert to WAV if not already in WAV format
        wav_filename = temp_filename
        if not temp_filename.lower().endswith('.wav'):
            status_text.text("Converting audio to WAV format (this may take a moment)...")
            sound = AudioSegment.from_file(temp_filename)
            wav_filename = temp_filename.replace(os.path.splitext(temp_filename)[1], '.wav')
            sound.export(wav_filename, format="wav")
        
        # Get audio duration for better progress estimation
        try:
            audio = AudioSegment.from_file(wav_filename)
            duration_seconds = len(audio) / 1000  # pydub returns duration in milliseconds
            duration_minutes = duration_seconds / 60
            
            status_text.text(f"Audio duration: {duration_minutes:.1f} minutes. Starting transcription...")
            progress_bar.progress(15)
        except:
            # If we can't determine the duration, continue anyway
            status_text.text("Starting transcription...")
            progress_bar.progress(15)
        
        # Transcribe the audio using Whisper
        status_text.text("Transcribing audio with Whisper (this may take several minutes)...")
        transcription = transcribe_audio_with_whisper(wav_filename, progress_bar)
        
        if transcription:
            progress_bar.progress(60)
            status_text.text("Processing meeting notes with AI...")
            
            st.session_state.transcription = transcription
            
            # Extract meeting notes using AI
            meeting_notes = extract_meeting_notes(transcription, model_provider, selected_model)
            
            if meeting_notes:
                progress_bar.progress(90)
                status_text.text("Displaying results...")
                st.session_state.meeting_notes = meeting_notes
            else:
                st.error("Failed to extract meeting notes.")
        else:
            st.error("Failed to transcribe audio.")
        
        # Clean up temp files
        try:
            os.unlink(temp_filename)
            if wav_filename != temp_filename:
                os.unlink(wav_filename)
        except:
            pass  # Ignore errors during cleanup
        
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        st.success("Meeting processing complete!")

# Display results if available
if st.session_state.transcription:
    st.header("Transcription")
    with st.expander("View Raw Transcription", expanded=False):
        st.text_area("Raw Transcription", st.session_state.transcription, height=300)
    
    if st.session_state.meeting_notes:
        st.header("Meeting Summary")
        
        # Display the meeting notes in a structured format
        if "raw_response" in st.session_state.meeting_notes:
            formatted_response = st.session_state.meeting_notes["raw_response"]
            
            # Try to parse and structure the response
            if "## Key Meeting Points" in formatted_response:
                # Split the response into sections
                sections = formatted_response.split("## ")
                
                # Create tabs for different sections
                tab1, tab2, tab3, tab4 = st.tabs(["📝 Meeting Points", "✅ Action Items", "⚖️ Decisions", "➡️ Next Steps"])
                
                for section in sections:
                    section = section.strip()
                    if section.startswith("Key Meeting Points"):
                        content = section.replace("Key Meeting Points", "").strip()
                        with tab1:
                            st.markdown(content if content else "No key meeting points identified.")
                    
                    elif section.startswith("Action Items"):
                        content = section.replace("Action Items", "").strip()
                        with tab2:
                            st.markdown(content if content else "No action items identified.")
                    
                    elif section.startswith("Decisions Made"):
                        content = section.replace("Decisions Made", "").strip()
                        with tab3:
                            st.markdown(content if content else "No decisions identified.")
                    
                    elif section.startswith("Next Steps"):
                        content = section.replace("Next Steps", "").strip()
                        with tab4:
                            st.markdown(content if content else "No next steps identified.")
            else:
                # If the response doesn't have the expected format, display as is
                st.subheader("Meeting Summary")
                st.markdown(formatted_response)
        
        # Add a download button for the meeting notes
        st.download_button(
            label="Download Meeting Notes",
            data=st.session_state.meeting_notes.get("raw_response", ""),
            file_name="meeting_notes.txt",
            mime="text/plain"
        )
        
        # Add option to copy notes to clipboard (simulated)
        if st.button("Copy Notes to Clipboard"):
            st.success("Notes copied to clipboard!")
            st.info("Note: Actual clipboard functionality requires JavaScript, which is not available in this Streamlit environment. The text is shown above for manual copying.")