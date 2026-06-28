# 📄 RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) application that lets users upload any PDF and ask natural language questions about its content.

## 🛠️ Tech Stack
- **LLM:** Groq (Llama 3.1 8B)
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
- **Vector Store:** FAISS
- **Framework:** LangChain
- **UI:** Streamlit

## 🔍 How it Works
1. PDF is loaded and split into text chunks
2. Chunks are embedded and stored in a FAISS vector database
3. User question is matched against chunks using semantic search
4. Relevant chunks + question are sent to Groq LLM for a grounded answer

## ⚙️ Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Groq API key to `.env`: `GROQ_API_KEY=your_key`
4. Run: `streamlit run app.py`

🔗 Live Demo: https://rag-pdf-chatbot-inbvlhwfpd6ply6pe7f469.streamlit.app