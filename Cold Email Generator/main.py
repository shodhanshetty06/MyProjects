import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chain import Chain
from portfolio import Portfolio
from utils import clean_text

# Function to reset input
def clear_input():
    st.session_state["url_input"] = ""
    st.session_state["email_generated"] = False
    st.session_state["generated_email"] = ""

# creation of app
def create_streamlit_app(llm, portfolio, clean_text):
    if st.session_state.get("email_generated", False):
        # title of generated email page
        st.title("📧 Here is your Cold Email")
        st.code(st.session_state["generated_email"], language='markdown')
        #to clear inputs given
        if st.button("Clear"):
            clear_input()

    else:
        st.title("📧 Cold Mail Generator")
        #inside box and outside box headding
        url_input = st.text_input(
            "Enter Your Job Posting Link Here",
            value=st.session_state.get("url_input", ""),
            placeholder="Enter the URL of the job posting"
        )
        #button allignment
        col1, col2, _ = st.columns([1, 1, 5])
        with col1:
            generate_button = st.button("Generate")
        with col2:
            clear_button = st.button("Clear")

        # Process After clicking generate button
        if generate_button:
            if not url_input:
                st.error("Please enter a URL to generate a cold email.")
            else:
                try:
                    loader = WebBaseLoader([url_input])
                    data = clean_text(loader.load().pop().page_content)
                    portfolio.load_portfolio()
                    jobs = llm.extract_jobs(data)
                    generated_email = ""
                    for job in jobs:
                        skills = job.get('skills', [])
                        links = portfolio.query_links(skills)
                        email = llm.write_mail(job, links)
                        generated_email += email + "\n\n"

                    # Save the generated email in session state
                    st.session_state["generated_email"] = generated_email
                    st.session_state["email_generated"] = True
                except Exception as e:
                    st.error(f"An Error Occurred: {e}")

        # Clear button
        if clear_button:
            clear_input()


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    chain = Chain()
    portfolio = Portfolio()
    if "url_input" not in st.session_state:
        st.session_state["url_input"] = ""
    if "email_generated" not in st.session_state:
        st.session_state["email_generated"] = False
    if "generated_email" not in st.session_state:
        st.session_state["generated_email"] = ""

    # Run the Streamlit app
    create_streamlit_app(chain, portfolio, clean_text)
