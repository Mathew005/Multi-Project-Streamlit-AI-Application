# In your main_app.py file

import streamlit as st

# Configure the page settings for the ENTIRE application here
st.set_page_config(
    page_title="Multi-Project Streamlit App",
    page_icon="🎉",
    layout="wide"  # Set the layout to wide for all pages
)

st.title("🎉 Welcome to the Multi-Project Streamlit App!")

st.sidebar.success("Select a project above.")

st.markdown(
    """
    This is a template for a multi-page Streamlit application.
    You can add different projects to the `pages` directory, and they will appear in the sidebar for navigation.

    **👈 Select a project from the sidebar** to see it in action!

    ### Available Projects:
    - **LLM Chatbot:** Chat with local or cloud-based LLMs, with document upload capabilities.
    - **Document Summarizer:** Upload a PDF or text file and generate a detailed summary.
    """
)