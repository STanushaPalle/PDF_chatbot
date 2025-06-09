# 🧠 Streamlit PDF RAG Application

A **privacy-first**, **local LLM-powered** application to chat with, summarize, and compare PDFs using:

- 🔍 **LangChain** for Retrieval-Augmented Generation (RAG)
- 💬 **Ollama** for local large language model (LLM) inference
- 📄 **Streamlit** for an interactive, browser-based UI
- 📚 **ChromaDB** for persistent vector storage

---

## 📦 Features

- Upload or select sample PDFs
- Extract and chunk text from documents
- Generate local vector embeddings using `nomic-embed-text`
- Store/retrieve content via ChromaDB vector store
- Chat with PDFs using a LangChain + ChatOllama RAG pipeline
- Summarize document content
- Compare two PDFs across tone, purpose, and structure
- 100% local — no external API or cloud required

---

## 🖼️ Architecture


```text
User Interface (Streamlit)
        |
        v
PDF Upload/Selection (sample or user-provided)
        |
        v
PDF Processing (pdfplumber + UnstructuredPDFLoader)
        |
        v
Text Chunking (LangChain RecursiveCharacterTextSplitter)
        |
        v
Embeddings Generation (Ollama + nomic-embed-text)
        |
        v
Vector Store (ChromaDB, with persistent directory)
        |
        v
Retrieval and MultiQuery Optimization (LangChain Retriever)
        |
        v
Chat + Summary + Comparison Chains (ChatOllama + Prompt Templates)
        |
        v
Chat UI + Summary UI + Compare UI (Streamlit containers and widgets)
```

---
## Project Structure
```
ollama_pdf_rag/
├── src/                      # Source code
│   ├── app/                  # Streamlit application
│   │   ├── components/       # UI components
│   │   │   ├── chat.py      # Chat interface
│   │   │   ├── pdf_viewer.py # PDF display
│   │   │   └── sidebar.py   # Sidebar controls
│   │   └── main.py          # Main app
│   └── core/                 # Core functionality
│       ├── document.py       # Document processing
│       ├── embeddings.py     # Vector embeddings
│       ├── llm.py           # LLM setup
│       └── rag.py           # RAG pipeline
├── data/                     # Data storage
│   ├── pdfs/                # PDF storage
│   │   └── sample/          # Sample PDFs
│   └── vectors/             # Vector DB storage
├── notebooks/               # Jupyter notebooks
│   └── experiments/         # Experimental notebooks
├── tests/                   # Unit tests
├── docs/                    # Documentation
└── run.py                   # Application runner
```

## 🚀 Getting Started

### Prerequisites

1. **Install Ollama**
   - Visit [Ollama's website](https://ollama.ai) to download and install
   - Pull required models:
     ```bash
     ollama pull llama3.2  # or your preferred model
     ollama pull nomic-embed-text
     ```

2. **Clone Repository**
   ```bash
   git clone https://github.com/STanushaPalle/PDF_chatbot
   cd PDF_chatbot
   ```

3. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

   Key dependencies and their versions:
   ```txt
   ollama==0.4.4
   streamlit==1.40.0
   pdfplumber==0.11.4
   langchain==0.1.20
   langchain-core==0.1.53
   langchain-ollama==0.0.2
   chromadb==0.4.22
   ```

### 🎮 Running the Application

#### Option 1: Streamlit Interface
```bash
python run.py
```
Then open your browser to `http://localhost:8501`

![Streamlit UI](st_app_ui.png)
*Streamlit interface showing PDF viewer and chat functionality*

#### Option 2: Jupyter Notebook
```bash
jupyter notebook
```
Open `updated_rag_notebook.ipynb` to experiment with the code

## ⚙️ Configuration

By default, the app connects to:

- `http://localhost:11434` for **Ollama**
- A persistent `chroma/` directory for **vectorstore**

You can configure these in a `.env` file (optional):

```env
OLLAMA_HOST=http://localhost:11434
VECTOR_DIR=chroma
```

# 📝 Project Overview

## ❗ Known Issues

| Challenge              | Solution                                          |
|------------------------|---------------------------------------------------|
| Ollama not running     | Ensure Ollama is active at `localhost:11434`     |
| Slow inference         | Use smaller/faster LLMs (e.g., `phi3`)           |
| UI bugs or freeze      | Clear session state or restart the app           |
| PDF with no text       | Fallback to OCR or `UnstructuredPDFLoader`       |

## 🔮 Future Roadmap

- ✅ Memory-enabled chat  
- 🧠 Metadata-based retrieval filters  
- 📊 Table/image-aware chunking  
- 🔄 Cross-PDF querying  
- 🛡️ User roles + secure document upload  

## 📜 License

MIT License – Open source, free to use and modify.

## 🤝 Acknowledgements

- [Ollama](https://ollama.com)
- [LangChain](https://www.langchain.com)
- [Streamlit](https://streamlit.io)
- [ChromaDB](https://www.trychroma.com)
- [nomic-embed-text](https://github.com/nomic-ai/nomic)

