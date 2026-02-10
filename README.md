# Resume Review Tool

A privacy-first resume screening application that uses local LLMs to analyze and rank job candidates.

## Features

- **Local Processing**: All analysis happens on your machine using Ollama - no cloud APIs
- **Multi-factor Scoring**: Ranks candidates using:
  - Keyword matching (must-haves vs nice-to-haves)
  - Semantic similarity (embedding-based comparison)
  - LLM fit analysis (0-100 score with rationale)
  - Seniority alignment (Junior/Experienced/Senior/Leadership)
- **Q&A Interface**: Ask specific questions about individual candidates
- **Export Results**: Download rankings as CSV

## Requirements

- Python 3.8+
- Ollama (local LLM runtime)

## Installation

1. **Install Ollama**
   - Download from [https://ollama.ai](https://ollama.ai)
   - Install and verify: `ollama --version`

2. **Download AI Models**
   ```bash
   ollama pull qwen3:8b
   ollama pull nomic-embed-text
   ```

3. **Install Python Dependencies**
   ```bash
   pip install streamlit pypdf docx2txt faiss-cpu numpy pandas requests
   ```

## Usage

1. **Start the application**
   ```bash
   python -m streamlit run ResumeToolLLLM.py
   ```

2. **Upload files**
   - Job description (TXT, PDF, or DOCX)
   - Multiple resumes (PDF, DOCX, or TXT)

3. **Configure scoring**
   - Set expected seniority level
   - Add must-have keywords (weight: 5)
   - Add nice-to-have keywords (weight: 3)
   - Adjust scoring weights in sidebar

4. **Analyze**
   - Click "Build Index & Rank Candidates"
   - Review ranked results
   - Ask questions about specific candidates

## Configuration

Customize via environment variables:
- `OLLAMA_BASE_URL`: Ollama API endpoint (default: `http://localhost:11434`)
- `LLM_MODEL`: Language model to use (default: `qwen3:8b`)
- `EMBED_MODEL`: Embedding model (default: `nomic-embed-text`)

## Privacy

All processing occurs locally on your machine. No resume data or job descriptions are sent to external services.

## License

[Add your license here]
