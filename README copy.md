# Installation Guide

## Install Ollama

1. [Ollama's website](https://ollama.ai) - download and install the application
2. After installation, pull the required models:
   ```bash
   ollama pull llama3.2  
   ollama pull llava:7b  
   ollama pull Zephyr:7b  
   ollama pull mistral:latest
   ollama pull deepseek-r1:7b
   ollama pull gemma 3:1b
   ollama pull nomic-embed-text
   ```

## Installing Ollama PDF Chatbot

1. Clone repository:
   ```bash
   git clone https://github.com/STanushaPalle/PDF_chatbot
   cd PDF_chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Verifying Installation

1. Start Ollama in the background
2. Run the application:
   ```bash
   python run.py
   #or
   #python -m streamlit run run.py
   ```
3. Open your browser to `http://localhost:8501`

## Troubleshooting

#### ONNX DLL Error
If you see this error:
```
DLL load failed while importing onnx_copy2py_export: a dynamic link Library (DLL) initialization routine failed.
```

Try these solutions:

1. Install Microsoft Visual C++ Redistributable:
   - Download both x64 and x86 versions from [Microsoft's website](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
   - Restart your computer

2. Or reinstall ONNX Runtime:
   ```bash
   pip uninstall onnxruntime onnxruntime-gpu
   pip install onnxruntime
   ```
