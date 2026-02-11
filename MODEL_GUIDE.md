# Model Selection Guide

This guide helps you choose the right AI models based on your hardware and quality needs.

## Your System Specs
- **CPU**: AMD Ryzen 7 9800X3D (8-core, 4.70 GHz)
- **RAM**: 64 GB
- **GPU**: NVIDIA GeForce RTX 5070 Ti (16GB VRAM)

## Language Models (LLM)

These models analyze resumes, score candidates, and answer questions.

### Recommended for Your System: **Qwen 2.5 32B** ⭐

| Model | Quality | Speed | VRAM | Best For |
|-------|---------|-------|------|----------|
| **qwen3:8b** | Good | Very Fast | ~5GB | Quick testing, multiple concurrent runs |
| **qwen2.5:14b** | Better | Fast | ~9GB | Balanced performance |
| **qwen2.5:32b** ⭐ | Excellent | Fast | ~18GB | **Recommended for your GPU** - Best quality/speed balance |
| **qwen2.5:72b** | Outstanding | Moderate | 40GB+ | Maximum quality (uses CPU+GPU offloading) |
| **llama3.3:70b** | Outstanding | Moderate | 40GB+ | Alternative to Qwen 72B |

### Installation

```bash
# Install recommended model
ollama pull qwen2.5:32b

# Or try the 72B for maximum quality
ollama pull qwen2.5:72b

# Keep the 8B for quick tests
ollama pull qwen3:8b
```

## Embedding Models

These models create semantic representations of text for similarity matching.

| Model | Quality | Speed | Best For |
|-------|---------|-------|----------|
| **nomic-embed-text** | Good | Very Fast | Default choice, efficient |
| **mxbai-embed-large** ⭐ | Better | Fast | **Recommended upgrade** - Better semantic understanding |
| **bge-large** | Better | Fast | Alternative high-quality option |

### Installation

```bash
# Recommended upgrade
ollama pull mxbai-embed-large

# Keep default as backup
ollama pull nomic-embed-text
```

## Quality Impact by Component

Different models affect different aspects of the screening:

| Component | What It Does | Affected By |
|-----------|--------------|-------------|
| **Keyword Score** | Counts keyword occurrences | Not affected by models |
| **Semantic Score** | Measures resume-job similarity | Embedding model |
| **LLM Fit Score** | Deep analysis of candidate fit | Language model (biggest impact) |
| **Seniority Assessment** | Estimates experience level | Language model |
| **Q&A Answers** | Responds to specific questions | Both models |

## Switching Models in the App

1. **Start the application**:
   ```bash
   python -m streamlit run ResumeToolLLLM.py
   ```

2. **In the sidebar**:
   - Select your preferred LLM from the "Language Model" dropdown
   - Select your preferred embedding model from "Embedding Model" dropdown
   - Click "📋 Check Model Availability" to verify they're installed

3. **If a model isn't installed**:
   - The app will show you the exact command to run
   - Copy and paste it into your terminal
   - Refresh the app

## Performance Tips

### For Maximum Quality (Your System Can Handle This!)
- **LLM**: `qwen2.5:32b` or `qwen2.5:72b`
- **Embedding**: `mxbai-embed-large`
- Expected processing time: 10-30 seconds per resume

### For Maximum Speed
- **LLM**: `qwen3:8b`
- **Embedding**: `nomic-embed-text`
- Expected processing time: 3-10 seconds per resume

### For Batch Processing (10+ resumes)
- **LLM**: `qwen2.5:32b` (best balance)
- **Embedding**: `nomic-embed-text` (faster without much quality loss)

## Environment Variables (Alternative Configuration)

Instead of using the UI dropdowns, you can set defaults via environment variables:

**Windows (PowerShell)**:
```powershell
$env:LLM_MODEL="qwen2.5:32b"
$env:EMBED_MODEL="mxbai-embed-large"
python -m streamlit run ResumeToolLLLM.py
```

**Windows (Command Prompt)**:
```cmd
set LLM_MODEL=qwen2.5:32b
set EMBED_MODEL=mxbai-embed-large
python -m streamlit run ResumeToolLLLM.py
```

## Troubleshooting

### Model Not Found Error
```bash
# Pull the model first
ollama pull <model-name>

# Verify it's installed
ollama list
```

### Out of Memory Error
- Switch to a smaller model (e.g., from 72B to 32B)
- Close other GPU-intensive applications
- Your 64GB RAM should handle any overflow

### Slow Performance
- Check if Ollama is using GPU: Task Manager → Performance → GPU
- Ensure no other applications are using the GPU heavily
- Try a smaller model for faster iterations

## My Recommendation for You

Based on your RTX 5070 Ti with 16GB VRAM:

**Best Setup**:
- **LLM**: `qwen2.5:32b` - Fits perfectly in your VRAM, excellent quality
- **Embedding**: `mxbai-embed-large` - Better semantic matching

This will give you professional-grade resume screening with great performance!
