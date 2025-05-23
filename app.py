import streamlit as st
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import os

st.title("Resume Matcher")

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Store resumes in memory
if "resumes" not in st.session_state:
    st.session_state.resumes = []

st.header("Upload Resumes (PDF Only)")
files = st.file_uploader("Choose resumes", type="pdf", accept_multiple_files=True)

for file in files:
    text = extract_text(file)
    st.session_state.resumes.append((file.name, text))

st.header("Paste Job Description")
jd = st.text_area("Job description goes here...")

if st.button("Find Best Matches") and jd:
    jd_embed = model.encode(jd, convert_to_tensor=True)
    results = []
    for name, resume_text in st.session_state.resumes:
        resume_embed = model.encode(resume_text, convert_to_tensor=True)
        score = util.cos_sim(jd_embed, resume_embed)[0][0].item()
        results.append((name, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    st.subheader("Top Matches:")
    for name, score in results:
        st.write(f"**{name}** — Match Score: `{score:.2f}`")
