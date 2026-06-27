import streamlit as st
import tempfile
import os
from rag_engine import build_qa_chain

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")
st.title("📄 RAG PDF Chatbot")
st.write("Upload a PDF and ask questions about it!")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing PDF... please wait"):
        qa_chain = build_qa_chain(tmp_path)
    
    st.success("PDF processed! Ask your questions below.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask something about the PDF...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):
           answer = qa_chain.invoke(question)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

    os.unlink(tmp_path)