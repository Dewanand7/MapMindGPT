# Local Setup Guide (Step-by-Step)

## Prerequisites

Install:

- Python 3.10 or 3.11
- Git
- Ollama (optional, for external LLMs)
- VS Code (recommended)

Recommended hardware:

### Minimum
- 8 GB RAM
- CPU

### Recommended
- 16+ GB RAM
- NVIDIA GPU (CUDA support)

---

# 1. Clone Repository

```bash
git clone https://github.com/Dewanand7/MapMindGPT.git
cd MapMindGPT
```

---

# 2. Create Virtual Environment

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then retry:

```powershell
.\.venv\Scripts\Activate.ps1
```

Expected:

```text
(.venv) PS C:\MapMindGPT>
```

---

## Linux / Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

# 3. Install Python Dependencies

```bash
pip install --upgrade pip
```

Install packages:

```bash
pip install torch torchvision torchaudio
pip install streamlit
pip install faiss-cpu
pip install sentence-transformers
pip install tokenizers
pip install ollama
pip install pypdf
pip install python-docx
pip install lxml
pip install openpyxl
pip install PyYAML
```

Supported document types:

```text
.txt .md .markdown .log .edi .x12 .dat
.pdf .docx .xlsx .csv .json .xml .html .htm .yaml .yml
```

---

# 4. Install Ollama (Optional)

Required only for:

- Qwen
- Llama
- Mistral
- DeepSeek
- CodeLlama

Download:

https://ollama.com

Check:

```bash
ollama --version
```

Pull models:

```bash
ollama pull qwen2.5:7b-instruct
ollama pull llama3:8b
ollama pull mistral:7b
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
```

---

# 5. Build Knowledge Corpus

If using built-in documents:

```bash
python ingest_docs.py
```

Expected:

```text
Corpus built successfully
```

---

# 6. Train Tokenizer

```bash
python tokenizer/train_tokenizer.py
```

Generates:

```text
tokenizer/vocab.json
tokenizer/merges.txt
```

---

# 7. Train Custom Model (Optional)

For better custom-model answers, first build the instruction-style corpus:

```bash
python build_instruction_dataset.py
```

If you want your own model:

```bash
python train.py
```

Generates:

```text
checkpoints/model.pt
```

You can run a small keyword-based evaluation after training:

```bash
python eval_custom_model.py
```

The Streamlit sidebar also includes **Custom Model Training** controls to build the
instruction dataset, train the custom model, and run evaluation from the UI.

Positive chat feedback can also be converted into extra training examples:

```bash
python feedback_to_instruction_dataset.py
```

When `data/feedback_instruction_corpus.txt` exists, `train.py` includes it automatically.

---

# 8. Build Vector Index

```bash
python build_index.py
```

Generates:

```text
data/vector.index
data/chunks.pkl
```

---

# 9. Run Streamlit App

```bash
streamlit run app.py
```

Expected:

```text
Local URL: http://localhost:8501
```

Open browser:

```text
http://localhost:8501
```

---

# 10. Select AI Model

Inside UI:

Choose:

### Custom
```text
MapMindGPT-Custom
```

Uses:

- your GPT model
- local checkpoint

Requires:

```text
checkpoints/model.pt
tokenizer files
```

---

### Ollama Models

Choose:

- qwen2.5
- llama3
- mistral
- deepseek
- codellama

Requires Ollama running.

---

# Upload Custom Documents

Supported:

- PDF
- DOCX
- TXT
- XML

Upload from sidebar.

MapMindGPT will:

1. save document
2. chunk text
3. embed text
4. rebuild vector DB
5. make knowledge searchable

---

# Common Issues

## Ollama not found

Error:

```text
ollama not recognized
```

Fix:

Install Ollama and restart terminal.

---

## Missing torch

Error:

```text
ModuleNotFoundError: torch
```

Fix:

```bash
pip install torch
```

---

## Missing tokenizer files

Error:

```text
tokenizer/vocab.json not found
```

Fix:

```bash
python tokenizer/train_tokenizer.py
```

---

## Missing model checkpoint

Error:

```text
checkpoints/model.pt not found
```

Fix:

```bash
python train.py
```

---

## Empty answers

Fix:

```bash
python build_index.py
```

---

# Fast Start (Already Trained)

If checkpoint + tokenizer already exist:

```bash
git clone https://github.com/Dewanand7/MapMindGPT.git
cd MapMindGPT

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

streamlit run app.py
```

