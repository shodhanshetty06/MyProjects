import re
import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Streamlit page configuration
st.set_page_config(page_title="News Odappa", page_icon="📰", layout="centered")

# Streamlit App UI
st.title("News Summarizer")
st.write("Summarize news articles effortlessly!")

# Initialize session state for reset functionality
if "reset" not in st.session_state:
    st.session_state["reset"] = False
if "summary_result" not in st.session_state:
    st.session_state["summary_result"] = ""
if "info_message" not in st.session_state:
    st.session_state["info_message"] = ""
if "input_url" not in st.session_state:
    st.session_state["input_url"] = ""

# Function to extract clean text from a news URL
def extract_article_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        # Extracting text from <p> tags
        paragraphs = soup.find_all("p")
        article_text = " ".join([p.get_text() for p in paragraphs if p.get_text()])
        return article_text.strip()  # Return the extracted text
    except Exception as e:
        st.error(f"Error fetching article content: {e}")
        return None

# Load the tokenizer and model
@st.cache_resource
# Cache the model loading to optimize app performance
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
    # Use CPU
    return summarizer

summarizer = load_model()

# Reset functionality
if st.session_state["reset"]:
    # Reset all session states
    st.session_state["input_url"] = ""
    st.session_state["summary_result"] = ""
    st.session_state["info_message"] = ""
    st.session_state["reset"] = False

# Input Field for News URL
news_url = st.text_input("Enter the news article URL:", value=st.session_state["input_url"], key="input_url")

# Summarize Button
if st.button("Summarize"):
    if not news_url:
        st.error("Please enter a valid news article URL.")
    else:
        st.session_state["info_message"] = "Fetching and summarizing the article..."
        article_content = extract_article_content(news_url)
        if article_content:
            try:
                # Perform summarization
                summary = summarizer(article_content, max_length=200, min_length=50, do_sample=False)
                st.session_state["summary_result"] = summary[0]['summary_text']
                st.session_state["info_message"] = ""
            except Exception as e:
                st.error(f"Error during summarization: {e}")

# Display info message or summary
if st.session_state["info_message"]:
    st.info(st.session_state["info_message"])
if st.session_state["summary_result"]:
    st.success("News Summary:")
    st.write(st.session_state["summary_result"])

# Reset Button
if st.button("Reset"):
    # Set reset state to True
    st.session_state["reset"] = True