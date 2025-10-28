import streamlit as st
import requests
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
import time
import hashlib
import sqlite3
import threading

# --- Page Config ---
load_dotenv()
st.set_page_config(page_title="News Summarizer", page_icon="📰", layout="wide")
st.title("📰 AI-Powered News Summarizer")
st.caption("Trending news, summarized for you by an AI model.")

# --- Load environment ---
load_dotenv()

# --- Constants ---
GNEWS_API_KEY = "dc56ca41bd5fd2c228bbf01b0afed940"  # Public API key provided by user
GNEWS_BASE_URL = "https://gnews.io/api/v4"

# --- Database Caching Functions ---
def get_db_connection():
    """Get a database connection for caching news and summaries"""
    conn = sqlite3.connect("news_cache.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_cache_db():
    """Initialize the cache database with required tables"""
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            params_hash TEXT,
            response_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS summary_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT UNIQUE,
            title TEXT,
            description TEXT,
            content TEXT,
            model_provider TEXT,
            selected_model TEXT,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Initialize the database
init_cache_db()

def fetch_news_cached(url, params):
    """Cached function to fetch news from GNews API using database"""
    # Create a hash of the parameters to use as cache key
    params_str = json.dumps(params, sort_keys=True)
    params_hash = hashlib.md5(params_str.encode()).hexdigest()
    
    conn = get_db_connection()
    
    # Check if we have a cached response
    cursor = conn.execute(
        "SELECT response_data FROM news_cache WHERE url = ? AND params_hash = ? AND expires_at > datetime('now')",
        (url, params_hash)
    )
    result = cursor.fetchone()
    
    if result:
        conn.close()
        return json.loads(result["response_data"])
    
    # No cached result, fetch from API
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    # Cache the result for 1 hour
    conn.execute(
        "INSERT OR REPLACE INTO news_cache (url, params_hash, response_data, expires_at) VALUES (?, ?, ?, datetime('now', '+1 hour'))",
        (url, params_hash, json.dumps(data))
    )
    conn.commit()
    conn.close()
    
    return data

def generate_summary_cached(title, description, content, model_provider, selected_model):
    """Cached function to generate AI summary using database with content-based key"""
    # Create a hash of the content to use as cache key
    content_hash = hashlib.md5(f"{title}{description}{content}{model_provider}{selected_model}".encode()).hexdigest()
    
    conn = get_db_connection()
    
    # Check if we have a cached summary
    cursor = conn.execute(
        "SELECT summary FROM summary_cache WHERE content_hash = ? AND expires_at > datetime('now')",
        (content_hash,)
    )
    result = cursor.fetchone()
    
    if result:
        conn.close()
        return result["summary"]
    
    # No cached result, generate summary
    summary = generate_summary(title, description, content, model_provider, selected_model)
    
    # Cache the result for 2 hours
    conn.execute(
        "INSERT OR REPLACE INTO summary_cache (content_hash, title, description, content, model_provider, selected_model, summary, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', '+2 hours'))",
        (content_hash, title, description, content, model_provider, selected_model, summary)
    )
    conn.commit()
    conn.close()
    
    return summary

def generate_summary(title, description, content, model_provider, selected_model):
    """Generate AI summary with rate limiting handling"""
    max_retries = 3
    retry_delay = 30  # seconds
    
    for attempt in range(max_retries):
        try:
            if model_provider == "Gemini":
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    return "Gemini API key not found"
                
                genai.configure(api_key=api_key)
                model_name = os.getenv("MODEL", "gemini-flash-latest")
                model = genai.GenerativeModel(model_name)
                
                prompt = f"""
                Summarize this news article in 2-3 sentences:
                Title: {title}
                Description: {description}
                Content: {content}
                """
                
                response = model.generate_content(prompt)
                return response.text if response.text else "Summary not available"
            
            elif model_provider == "Ollama":
                import requests as req
                ollama_url = "http://localhost:11434/api/generate"
                payload = {
                    "model": selected_model,
                    "prompt": f"Summarize this news article in 2-3 sentences: Title: {title}, Description: {description}, Content: {content}",
                    "stream": False
                }
                
                resp = req.post(ollama_url, json=payload)
                resp.raise_for_status()
                return resp.json().get("response", "Summary not available")
        
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                if attempt < max_retries - 1:  # Not the last attempt
                    # Show a warning and wait before retrying
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    st.warning(f"Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"Rate limit exceeded after {max_retries} attempts. Please try again later."
            else:
                return f"Error generating summary: {str(e)}"
    
    return "Error generating summary"

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # AI Provider Selection
    model_provider = st.selectbox("Choose an AI provider:", ["Gemini", "Ollama"])
    
    if model_provider == "Ollama":
        st.info("Ollama provider selected. Make sure Ollama is running locally.")
        selected_model = st.text_input("Enter Ollama model name", value="llama3")
    elif model_provider == "Gemini":
        if os.getenv("GEMINI_API_KEY"):
            st.success("Gemini API Key loaded from .env file.")
            selected_model = os.getenv("MODEL", "gemini-flash-latest")
        else:
            st.error("GEMINI_API_KEY not found in .env file.")
            st.stop()
    
    st.divider()
    
    # News Configuration
    st.header("📰 News Settings")
    
    # Category Selection
    categories = {
        "General": "general",
        "World": "world", 
        "Nation": "nation",
        "Business": "business",
        "Technology": "technology",
        "Entertainment": "entertainment",
        "Sports": "sports",
        "Science": "science",
        "Health": "health"
    }
    selected_category = st.selectbox("News Category", options=list(categories.keys()), index=0)
    
    # Language Selection
    languages = {
        "English": "en",
        "Spanish": "es", 
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
        "Chinese": "zh",
        "Japanese": "ja",
        "Arabic": "ar"
    }
    selected_language = st.selectbox("Language", options=list(languages.keys()), index=0)
    
    # Country Selection
    countries = {
        "United States": "us",
        "United Kingdom": "gb",
        "Canada": "ca",
        "Australia": "au",
        "India": "in",
        "South Africa": "za",
        "Singapore": "sg",
        "Ireland": "ie",
        "New Zealand": "nz",
        "France": "fr",
        "Germany": "de",
        "Japan": "jp",
        "Italy": "it",
        "Spain": "es",
        "Netherlands": "nl",
        "Brazil": "br",
        "Mexico": "mx",
        "South Korea": "kr",
        "Russia": "ru",
        "China": "cn",
        "Saudi Arabia": "sa",
        "Switzerland": "ch",
        "Poland": "pl"
    }
    selected_country = st.selectbox("Country", options=list(countries.keys()), index=0)
    
    # Number of articles
    num_articles = st.slider("Number of articles", min_value=1, max_value=20, value=10)
    
    # Date filter
    st.subheader("Date Filter")
    date_filter = st.radio("Select date range:", ("All time", "Today", "This week", "This month"))
    
    # Map date filter to API parameters
    if date_filter == "Today":
        from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    elif date_filter == "This week":
        from_date = (datetime.now() - timedelta(weeks=1)).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    elif date_filter == "This month":
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    else:
        from_date = None  # No date filter


# --- Main Page UI ---
st.header("Latest News")

# Create tabs for different functions
tab1, tab2 = st.tabs(["📰 Top Headlines", "🔍 Search News"])

with tab1:
    st.subheader("Top Headlines")
    
    if st.button("Fetch Top Headlines", type="primary"):
        with st.spinner("Fetching top headlines..."):
            try:
                # Build the API URL for top headlines
                api_url = f"{GNEWS_BASE_URL}/top-headlines"
                params = {
                    "category": categories[selected_category],
                    "apikey": GNEWS_API_KEY,
                    "lang": languages[selected_language],
                    "country": countries[selected_country],
                    "max": num_articles
                }
                
                # Add date filter if selected
                if from_date:
                    params["from"] = from_date
                
                # Use the cached function to fetch news
                data = fetch_news_cached(api_url, params)
                
                if "articles" in data and data["articles"]:
                    articles = data["articles"]
                    
                    st.success(f"Found {len(articles)} articles")
                    
                    # Display articles
                    for i, article in enumerate(articles):
                        with st.expander(f"📰 {article['title']}", expanded=False):
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                if article.get('image'):
                                    st.image(article['image'], use_container_width=True)
                                else:
                                    st.write("No image available")
                            
                            with col2:
                                st.write(f"**Source:** {article['source']['name']}")
                                st.write(f"**Published:** {article['publishedAt']}")
                                st.write(f"**Description:** {article['description']}")
                                
                                # Link to the original article
                                st.markdown(f"[Read full article]({article['url']})")
                                
                                # AI Summary section with caching and rate limiting
                                with st.spinner("Generating/Retrieving AI summary..."):
                                    # Use the cached summary function
                                    summary = generate_summary_cached(
                                        article['title'],
                                        article['description'],
                                        article.get('content', 'No content available'),
                                        model_provider,
                                        selected_model
                                    )
                                    
                                    st.subheader("AI Summary")
                                    st.write(summary)
                
                else:
                    st.warning("No articles found for the selected criteria.")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching news: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

with tab2:
    st.subheader("Search News")
    
    search_query = st.text_input("Enter search keywords", placeholder="e.g., technology, elections, sports...")
    
    if st.button("Search News"):
        if not search_query:
            st.warning("Please enter search keywords")
        else:
            with st.spinner("Searching news..."):
                try:
                    # Build the API URL for search
                    api_url = f"{GNEWS_BASE_URL}/search"
                    params = {
                        "q": search_query,
                        "apikey": GNEWS_API_KEY,
                        "lang": languages[selected_language],
                        "country": countries[selected_country],
                        "max": num_articles
                    }
                    
                    # Add date filter if selected
                    if from_date:
                        params["from"] = from_date
                    
                    # Use the cached function to fetch news
                    data = fetch_news_cached(api_url, params)
                    
                    if "articles" in data and data["articles"]:
                        articles = data["articles"]
                        
                        st.success(f"Found {len(articles)} articles for '{search_query}'")
                        
                        # Display articles
                        for i, article in enumerate(articles):
                            with st.expander(f"📰 {article['title']}", expanded=False):
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    if article.get('image'):
                                        st.image(article['image'], use_container_width=True)
                                    else:
                                        st.write("No image available")
                                
                                with col2:
                                    st.write(f"**Source:** {article['source']['name']}")
                                    st.write(f"**Published:** {article['publishedAt']}")
                                    st.write(f"**Description:** {article['description']}")
                                    
                                    # Link to the original article
                                    st.markdown(f"[Read full article]({article['url']})")
                                    
                                    # AI Summary section with caching and rate limiting
                                    with st.spinner("Generating/Retrieving AI summary..."):
                                        # Use the cached summary function
                                        summary = generate_summary_cached(
                                            article['title'],
                                            article['description'],
                                            article.get('content', 'No content available'),
                                            model_provider,
                                            selected_model
                                        )
                                        
                                        st.subheader("AI Summary")
                                        st.write(summary)
                    
                    else:
                        st.warning(f"No articles found for '{search_query}' with the selected criteria.")
                
                except requests.exceptions.RequestException as e:
                    st.error(f"Error searching news: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

# Info section
with st.expander("ℹ️ How to use this News Summarizer"):
    st.markdown("""
    - **Top Headlines**: Get the latest news based on the selected category, language, and country
    - **Search News**: Enter keywords to search for specific news articles
    - Articles are fetched from the GNews API and summarized using AI
    - You can choose between Gemini or Ollama for AI summarization
    """)