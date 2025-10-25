import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa import RetrievalQA
# from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI


# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = api_key


# Step 1: Read PDF
def load_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Step 2: Chunking
def chunking(doc):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents([doc])
    return chunks


# Step 3: Create Embeddings
def create_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    db = FAISS.from_documents(chunks, embeddings)
    return db


# Step 4: Ask Question (Restricted)
def ask(query, db, threshold=0.4):
    """Answer only if the question is related to the PDF content."""
    # Find most similar document chunks
    retriever = db.as_retriever(search_kwargs={"score_threshold": threshold})
    docs = db.similarity_search_with_score(query, k=1)

    if not docs or docs[0][1] < threshold:
        return "⚠️ Sorry, your question doesn’t seem to be related to the uploaded PDF."

    llm = ChatOpenAI(model='gpt-4o-mini', api_key=api_key, max_tokens=500)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    response = qa_chain.invoke({'query': query})
    return response['result']


# Streamlit UI
st.title("📘 PDF QnA Bot")
file = st.file_uploader("Upload a PDF", type=['pdf'])

if file:
    with st.spinner("Reading PDF..."):
        data = load_pdf(file)
        chunks = chunking(data)
        db = create_embeddings(chunks)
        st.success("✅ PDF processed successfully!")

    query = st.text_input("Ask a question about your PDF:")
    if query:
        with st.spinner("Thinking... 🤔"):
            answer = ask(query, db)
            st.text_area("Answer:", answer,height=600)
