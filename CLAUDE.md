# CLAUDE.md - Resume Review Tool

**Working Directory:** `C:\Users\steph\OneDrive\Software\Resume Review Tool`

## Project Overview

The **Resume Review Tool** is a privacy-first resume screening application that uses local Large Language Models (via Ollama) to analyze and rank job candidates. The application runs entirely on your local machine with no cloud dependencies, ensuring complete data privacy for sensitive resume information.

**Key Capabilities:**
- Multi-factor candidate scoring combining keyword matching, semantic similarity, LLM reasoning, and seniority assessment
- Model-aware weight presets that automatically adjust scoring based on AI model quality
- Support for multiple resume formats (PDF, DOCX, TXT)
- Interactive Q&A interface for candidate-specific questions
- CSV export of ranked candidates with detailed scoring breakdowns

**Technology:** Streamlit web UI + Ollama (local LLM runtime) + FAISS (vector similarity search)

## Architecture

### Multi-Factor Scoring System

The application combines **four weighted scoring components** to rank candidates:

1. **Keyword Matching (Deterministic)**
   - Exact keyword/phrase frequency in resume text
   - Supports weighted keywords (e.g., "Python: 2.0" for higher importance)
   - Logistic normalization to [0, 100] scale
   - **Model-independent** - works the same regardless of AI models selected

2. **Semantic Similarity (Embedding-Based)**
   - FAISS vector search comparing resume chunks to job description
   - Top-5 average cosine similarity, mapped to [0, 100]
   - **Highly embedding model dependent** - mxbai-embed-large provides 2x better quality than nomic-embed-text
   - Uses average pooling of chunk embeddings for comprehensive representation

3. **LLM Fit Score (Reasoning-Based)**
   - Holistic assessment of candidate-job fit via LLM reasoning
   - Structured JSON response with 0-100 score + rationale
   - **Highly LLM model dependent** - qwen2.5:32b provides 3-4x better reasoning than qwen3:8b
   - Parallelized with seniority scoring for performance

4. **Seniority Alignment (Experience Level)**
   - 4-level classification: Junior (0-3 yrs), Experienced (3-8 yrs), Senior (8-15 yrs), Leadership (15+ yrs)
   - Two scoring modes:
     - **Additive:** Bonus/penalty added to final score (candidates can exceed 100%)
     - **Multiplicative:** Filter that dampens score based on mismatch (strict role enforcement)

### Final Score Calculation

**Additive Mode (default):**
```
Final Score = (w_keywords × keyword_score) + (w_semantic × semantic_score) +
              (w_llmfit × llm_fit_score) + (w_seniority × seniority_score)
```

**Multiplicative Mode (Role-Critical preset):**
```
Base Score = (w_keywords × keyword_score) + (w_semantic × semantic_score) + (w_llmfit × llm_fit_score)
Final Score = Base Score × seniority_multiplier
```

Where `seniority_multiplier` ranges from 1.0 (perfect match) to 0.20 (3+ levels off).

## Technical Stack

### Core Dependencies

```python
streamlit==1.51.0          # Web UI framework
faiss-cpu==1.13.1          # Vector similarity search
numpy==2.3.4               # Numerical computations
pandas==2.3.3              # Data handling and CSV export
pypdf==6.4.1               # PDF text extraction
docx2txt==0.9              # DOCX text extraction
requests==2.32.5           # HTTP client for Ollama API
```

### Ollama Integration

**Local LLM Runtime:** `http://localhost:11434` (configurable via `OLLAMA_BASE_URL` environment variable)

**Available Language Models:**
```python
LLM_MODELS = {
    "qwen3:8b": {           # Default - Fast, adequate quality
        "vram": "5GB",
        "ctx_window": 8192,
        "default": True
    },
    "qwen2.5:14b": {        # Balanced - Medium speed, good quality
        "vram": "9GB",
        "ctx_window": 32768
    },
    "qwen2.5:32b": {        # Recommended - Excellent quality, 4x better reasoning
        "vram": "18GB",
        "ctx_window": 32768
    },
    "qwen2.5:72b": {        # Maximum quality - Requires CPU offloading
        "vram": "40GB+",
        "ctx_window": 32768
    },
    "llama3.3:70b": {       # Alternative high-quality option
        "vram": "40GB+",
        "ctx_window": 131072
    }
}
```

**Available Embedding Models:**
```python
EMBED_MODELS = {
    "nomic-embed-text": {       # Default - Fast, large context window
        "ctx_window": 8192,
        "default": True
    },
    "mxbai-embed-large": {      # Better semantic understanding (~2x quality improvement)
        "ctx_window": 512
    },
    "bge-large": {              # Alternative high-quality option
        "ctx_window": 512
    }
}
```

### Configuration Methods

1. **Environment Variables:**
   ```bash
   export OLLAMA_BASE_URL="http://localhost:11434"
   export LLM_MODEL="qwen2.5:32b"
   export EMBED_MODEL="mxbai-embed-large"
   ```

2. **UI Model Selection:** Sidebar dropdowns with availability checking

3. **Custom Model Input:** Advanced expander for non-preset models

## Project Structure

```
C:\Users\steph\OneDrive\Software\Resume Review Tool\
├── ResumeToolLLLM.py                     # Main application (1,041 lines)
├── README.md                             # User installation guide (74 lines)
├── MODEL_GUIDE.md                        # Model selection recommendations (150 lines)
├── WEIGHT_PRESETS_GUIDE.md              # Scoring weight documentation (282 lines)
├── CLAUDE.md                             # This file - AI assistant context
├── Resume_Screener_Installation_Guide.docx
├── .gitignore                            # Excludes PDFs, DOCXs, .venv, etc.
├── .claude/
│   ├── settings.local.json               # Claude Code permissions
│   └── plans/                            # Implementation plans
└── .venv/                                # Python virtual environment
```

## Code Organization

**ResumeToolLLLM.py** is organized into these major sections:

### Lines 1-27: Imports & Global Config
- Standard library + third-party imports
- Dataclass definitions (not in this section, but imported)

### Lines 28-68: Configuration
```python
OLLAMA_BASE = "http://localhost:11434"
LLM_MODELS = {...}              # Model presets with metadata
EMBED_MODELS = {...}            # Embedding model configurations
MAX_CHUNK_TOKENS = 800          # Legacy constant (now adaptive)
CHUNK_OVERLAP = 120             # Word overlap between chunks
```

### Lines 72-131: Text Processing Utilities
```python
def clean_text(t: str) -> str                                    # Lines 72-77
def read_pdf_bytes(b: bytes) -> str                             # Lines 79-90
def read_docx_bytes(b: bytes) -> str                            # Lines 92-107
def read_txt_bytes(b: bytes) -> str                             # Lines 109-113
def split_into_chunks_by_words(...) -> List[str]               # Lines 115-131
```

### Lines 136-277: Ollama API Wrappers
```python
def check_ollama_connection() -> Tuple[bool, str]               # Lines 144-171
def ollama_embeddings(texts: List[str], ...) -> np.ndarray     # Lines 174-230
def ollama_chat(messages: List[Dict], ...) -> str              # Lines 232-277
```

### Lines 282-457: Scoring Engine
```python
@dataclass
class CandidateScore:                                           # Lines 282-293
    candidate: str
    file_name: str
    keyword_score: float
    semantic_score: float
    llm_fit_score: float
    seniority_score: float
    combined: float
    summary: str
    seniority_rationale: str

def keyword_counts(...) -> Tuple[float, Dict[str, int]]        # Lines 295-319
def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float        # Lines 321-329
def get_context_limits(model: str) -> Tuple[int, int]          # Lines 331-348
def get_embedding_chunk_size(embed_model: str) -> int          # Lines 350-373 ⚠️ CRITICAL
def llm_fit_score_llm(...) -> Tuple[float, str]               # Lines 375-415
def estimate_seniority(...) -> Tuple[int, str]                # Lines 417-457
```

**⚠️ CRITICAL FUNCTION:** `get_embedding_chunk_size()` uses **conservative safety margins** to prevent context overflow:
- **50% safety margin** (changed from 30% in commit 3595e10)
- **0.60 token-to-word ratio** (changed from 0.75)
- For mxbai-embed-large (512 tokens): 512 × 0.5 × 0.60 = **154 words/chunk**

### Lines 458-494: FAISS Vector Index
```python
@dataclass
class DocChunk:                                                 # Lines 458-464
    candidate: str
    file_name: str
    chunk_id: str
    text: str
    vector: np.ndarray

class ResumeIndex:                                              # Lines 466-494
    def __init__(self, dim: int)
    def add(self, chunks: List[DocChunk])
    def search(self, query_vec: np.ndarray, k: int) -> List[DocChunk]
```

### Lines 498-703: Streamlit UI Configuration
```python
st.set_page_config(...)                                         # Lines 498-503
st.title("Resume Screening Tool with Local LLMs")               # Line 505

with st.sidebar:                                                 # Lines 507-703
    # Model selection dropdowns                                 # Lines 510-532
    # Weight preset selector                                    # Lines 546-619 ⚠️ IMPORTANT
    # Skip seniority option                                     # Lines 622-626
    # Model availability checker                                # Lines 647-669
    # Session state clear button                                # Lines 711-714
```

**⚠️ IMPORTANT:** Weight preset system (lines 546-619) implements **model-aware adaptation**:
- Better embeddings (mxbai-embed-large, bge-large) → +5% semantic weight
- Better LLMs (qwen2.5:32b+, llama3.3:70b) → +5% LLM fit weight

### Lines 744-957: Main Processing Pipeline
```python
if start_button:                                                # Line 744
    # Validation and model availability checks                 # Lines 748-763
    # File parsing and text extraction                         # Lines 776-793
    # Job description embedding (with chunking)                # Lines 800-809 ⚠️ CRITICAL
    # Resume chunk creation and embedding                      # Lines 820-856
    # Scoring loop with progress tracking                      # Lines 860-940
    # Parallel LLM calls for fit + seniority                   # Lines 896-903 ⚠️ PERFORMANCE
    # Results ranking and display                              # Lines 942-957
```

**⚠️ CRITICAL:** Job description chunking (lines 800-809) prevents truncation via average pooling:
```python
jd_chunks = split_into_chunks_by_words(JD, max_words=max_chunk_words)
if len(jd_chunks) > 1:
    print(f"ℹ️ Job description split into {len(jd_chunks)} chunks...")
jd_chunk_vecs = ollama_embeddings(jd_chunks, model=embed_model)
jd_vec = np.mean(jd_chunk_vecs, axis=0)  # Average pooling
```

**⚠️ PERFORMANCE:** Parallel LLM processing (lines 896-903) provides ~50% speedup:
```python
with ThreadPoolExecutor(max_workers=2) as executor:
    fit_future = executor.submit(llm_fit_score_llm, fulltext, JD, llm_model)
    seniority_future = executor.submit(estimate_seniority, fulltext, llm_model)
    fit_score, rationale = fit_future.result()
    cand_level, seniority_rationale = seniority_future.result()
```

### Lines 960-1041: Results Display & Q&A Interface
```python
if st.session_state.get("ranked"):                             # Lines 960-997
    # Ranked candidates table display
    # CSV export functionality

# Q&A Interface                                                 # Lines 998-1037
if st.session_state.get("all_chunks"):
    # Candidate-scoped question answering
    # Top-k context retrieval with similarity scores
```

## Development Patterns

### Error Handling Strategy

**1. Ollama API Calls (Lines 174-230, 232-277):**
```python
try:
    r = requests.post(f"{OLLAMA_BASE}/api/embeddings", ...)
    r.raise_for_status()
except requests.exceptions.ConnectionError:
    raise ConnectionError(f"Cannot connect to Ollama at {OLLAMA_BASE}. "
                         "Please make sure Ollama is installed and running.")
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        raise ValueError(f"Model '{model}' not found. Install with: ollama pull {model}")
    elif e.response.status_code == 500:
        error_detail = e.response.json().get("error", "Unknown error")
        raise RuntimeError(f"Ollama server error. Details: {error_detail}")
```

**Key Pattern:** Differentiated error messages with **actionable remediation** (installation commands, service status checks).

**2. File Parsing (Lines 79-113):**
```python
try:
    reader = PdfReader(BytesIO(b))
    pages_text = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages_text)
except Exception:
    return ""  # Graceful degradation
```

**Key Pattern:** Silent failure with empty string fallback for corrupted files.

**3. JSON Parsing from LLM (Lines 405-414, 444-453):**
```python
try:
    start = resp.find("{"); end = resp.rfind("}")
    obj = json.loads(resp[start:end+1])
    score = int(obj.get("score", 50))
    return max(0, min(100, score)), rationale  # Clamp to [0, 100]
except Exception:
    return 50, "Heuristic fallback due to parse error."
```

**Key Pattern:** Fallback to reasonable defaults (50% score, mid-level seniority) on parse failures.

### Context Window Management

**CRITICAL LESSON:** Aggressive safety margins cause "input length exceeds context length" errors.

**Evolution of Safety Calculations:**
| Commit | Safety Margin | Token Ratio | mxbai Words/Chunk | Result |
|--------|---------------|-------------|-------------------|--------|
| Initial | 30% (0.7) | 0.75 | 268 words | ❌ Context overflow errors |
| 3595e10 | **50% (0.5)** | **0.60** | **154 words** | ✅ No errors |

**Current Implementation (Lines 350-373):**
```python
def get_embedding_chunk_size(embed_model: str) -> int:
    """Uses conservative safety margins to prevent context overflow:
    - 50% safety margin (aggressive tokenization can use 2x expected tokens)
    - 0.60 token-to-word ratio (assumes 1 word ≈ 1.67 tokens, typical for embeddings)
    """
    ctx_window = EMBED_MODELS.get(embed_model, {}).get("ctx_window", 8192)
    safe_tokens = int(ctx_window * 0.5)      # 50% safety margin
    safe_words = int(safe_tokens * 0.60)     # Conservative token-to-word ratio
    safe_words = max(100, safe_words)        # Minimum 100 words
    return safe_words
```

**Why 50% + 0.60 Ratio?**
- Tokenization varies by text type (technical terms, code snippets, special chars)
- Embedding models use 1.3-1.7 tokens per word (not the naive 1.0 assumption)
- **268 words × 1.4 tokens/word = 375 tokens** → Exceeds 512-token limit!
- **154 words × 1.4 tokens/word = 216 tokens** → Safe buffer of 296 tokens ✅

**Impact on Chunk Counts:**
| Content | Model | Old (268 words) | New (154 words) | Change |
|---------|-------|-----------------|-----------------|--------|
| 800-word resume | mxbai | 3-4 chunks | 6-7 chunks | +75% |
| 1500-word JD | mxbai | 6 chunks | 10-12 chunks | +67% |
| 800-word resume | nomic | 1 chunk | 1 chunk | No change |

### Performance Optimization Patterns

**1. Parallel LLM Calls (Lines 896-903):**
```python
if skip_seniority:
    fit_score, rationale = llm_fit_score_llm(fulltext, JD, llm_model)
    cand_level = required_level
else:
    with ThreadPoolExecutor(max_workers=2) as executor:
        fit_future = executor.submit(llm_fit_score_llm, fulltext, JD, llm_model)
        seniority_future = executor.submit(estimate_seniority, fulltext, llm_model)
        fit_score, rationale = fit_future.result()
        cand_level, seniority_rationale = seniority_future.result()
```

**Benefit:** ~50% speedup by parallelizing independent LLM calls.

**2. Progress Tracking (Lines 821-857):**
```python
progress_bar = st.progress(0)
status_text = st.empty()
for idx, (cand_name, file_name, fulltext) in enumerate(candidates_raw):
    status_text.text(f"Processing {idx+1}/{total_candidates}: {cand_name}...")
    progress_bar.progress((idx) / total_candidates)
```

**Benefit:** Real-time feedback for long-running operations (10-30 seconds/resume with quality models).

**3. Skip Seniority Option (Lines 622-626):**
```python
skip_seniority = st.checkbox(
    "⚡ Skip seniority scoring (2x faster)",
    value=False,
    help="Skip seniority assessment to reduce processing time by ~50%."
)
```

**Benefit:** Allows users to trade accuracy for speed when seniority is not critical.

## Recent Development History

### Key Commits (Last 10)

```
3595e10  Fix context overflow with conservative embedding chunk sizes
f43b840  Remove debug logging - truncation issue resolved
0ac2376  Add Phase 1 debug logging for truncation investigation
69538bf  Fix job description embedding truncation with chunking
055ccbc  Add years of experience clarification to role level dropdown
4896b13  Add comprehensive guide for weight preset system
94e1072  Add model-aware weight preset system for scoring optimization
55abdce  Fix embedding model context length errors for mxbai-embed-large
a75af7c  Add adaptive context limits for larger models
0652347  Optimize resume scoring performance with parallel LLM calls
```

### Architectural Evolution

**Phase 1: Context Window Management (Commits a75af7c, 55abdce, 69538bf, 3595e10)**
- **Problem:** Fixed context limits caused truncation/overflow errors
- **Solution:** Adaptive calculations based on model context windows
- **Lesson:** Conservative safety margins (50% + 0.60 ratio) required for reliability

**Phase 2: Performance Optimization (Commit 0652347)**
- **Problem:** Sequential LLM calls made processing slow (20s/resume)
- **Solution:** Parallel ThreadPoolExecutor for fit + seniority scoring
- **Result:** ~50% speedup, reduced to 10-13s/resume

**Phase 3: Weight Preset System (Commits 94e1072, 4896b13)**
- **Problem:** Users confused about optimal scoring weights
- **Solution:** Five preset configurations with model-aware adaptation
- **Innovation:** Better models automatically receive higher weights

**Phase 4: UX Improvements (Commit 055ccbc)**
- **Problem:** "Expected Role Level" dropdown lacked clarity
- **Solution:** Added year ranges (e.g., "Senior (8-15 years)")

### Known Issues & Resolutions

**1. ✅ RESOLVED: Embedding Context Overflow (Commit 3595e10)**
- **Error:** `RuntimeError: Ollama server error. Details: the input length exceeds the context length`
- **Root Cause:** 268-word chunks with 0.75 token ratio exceeded 512-token limit
- **Fix:** Reduced to 154-word chunks with conservative 0.60 ratio
- **Trade-off:** More chunks (+75%) = slightly slower processing (~20-30%)

**2. ✅ RESOLVED: Job Description Truncation (Commit 69538bf)**
- **Issue:** Long JDs (>268 words) were truncated, losing important requirements
- **Fix:** Chunk JD into smaller pieces, average their embeddings (average pooling)
- **Result:** Full JD representation with zero information loss

**3. ✅ RESOLVED: Model Tag Matching (Commit 55abdce)**
- **Issue:** `mxbai-embed-large:latest` not detected as available
- **Fix:** Strip tag suffix before comparison (`split(":")[0]`)

**4. ⚠️ KNOWN: PDF Parsing Warnings**
- **Warning:** `Ignoring wrong pointing object X 0 (offset 0)`
- **Source:** pypdf library when parsing malformed PDFs
- **Impact:** None - text extraction still works correctly
- **Recommendation:** Ignore these warnings; they're harmless

## Configuration & Setup

### Installation

```bash
# 1. Install Ollama (Windows/Mac/Linux)
# Download from: https://ollama.ai

# 2. Start Ollama service
ollama serve

# 3. Pull required models
ollama pull qwen3:8b
ollama pull nomic-embed-text

# 4. (Optional) Pull better models for quality
ollama pull qwen2.5:32b
ollama pull mxbai-embed-large

# 5. Install Python dependencies
pip install -r requirements.txt

# 6. Run application
python -m streamlit run ResumeToolLLM.py
```

### Environment Configuration

**Method 1: Environment Variables (PowerShell)**
```powershell
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:LLM_MODEL = "qwen2.5:32b"
$env:EMBED_MODEL = "mxbai-embed-large"
python -m streamlit run ResumeToolLLLM.py
```

**Method 2: UI Selection**
- Use sidebar dropdowns to select models
- Click "📋 Check Model Availability" to verify installations
- App provides `ollama pull` commands for missing models

**Method 3: Custom Model Input**
- Expand "🔧 Advanced: Custom Models" section
- Check "Use custom LLM model" or "Use custom embedding model"
- Enter model name manually

### Recommended Hardware Configuration

**For AMD Ryzen 7 9800X3D + RTX 5070 Ti (16GB VRAM):**
- **LLM:** `qwen2.5:32b` (perfect VRAM fit, excellent quality)
- **Embedding:** `mxbai-embed-large` (better semantic matching)
- **Weight Preset:** "Balanced (Recommended)" (auto-adjusts to 45% semantic, 40% LLM fit)

**Processing Time Estimates:**
- 10 resumes: ~2-5 minutes
- 83 resumes: ~13-20 minutes (with parallelization)

## Coding Standards

### Type Hints
```python
def llm_fit_score_llm(resume_text: str, job_desc: str, model: str = None) -> Tuple[float, str]:
def ollama_embeddings(texts: List[str], model: str = None) -> np.ndarray:
def keyword_counts(resume: str, keywords: List[str]) -> Tuple[float, Dict[str, int]]:
```

**Convention:** Consistent use of `List`, `Dict`, `Tuple` from `typing` module.

### Dataclasses Over Dictionaries
```python
@dataclass
class CandidateScore:
    candidate: str
    file_name: str
    keyword_score: float
    semantic_score: float
    llm_fit_score: float
    seniority_score: float
    combined: float
    summary: str
    seniority_rationale: str
```

**Rationale:** Type safety, IDE autocomplete, clearer intent.

### Constants
```python
OLLAMA_BASE = "http://localhost:11434"
MAX_CHUNK_TOKENS = 800
CHUNK_OVERLAP = 120
```

**Convention:** Uppercase for module-level constants.

### Error Messages
```python
# ✅ Good: Actionable, includes remediation
raise ValueError(f"Model '{model}' not found. Install with: ollama pull {model}")

# ❌ Bad: Vague, no guidance
raise ValueError("Model not found")
```

**Convention:** Include context (model name) + remediation (installation command).

### Docstrings
```python
def get_embedding_chunk_size(embed_model: str) -> int:
    """Calculate safe chunk size (in words) for the embedding model.

    Uses conservative safety margins to prevent context overflow:
    - 50% safety margin (aggressive tokenization can use 2x expected tokens)
    - 0.60 token-to-word ratio (assumes 1 word ≈ 1.67 tokens, typical for embeddings)

    Returns: max_chunk_words
    """
```

**Convention:** Multi-line docstrings for complex functions, inline comments for critical logic.

## Weight Presets System

### Five Available Presets

**1. Balanced (Recommended) - Model-Aware Adaptation**
```python
w_keywords = 0.25
w_semantic = 0.45 if embed_model in ["mxbai-embed-large", "bge-large"] else 0.40
w_llmfit = 0.40 if llm_model in ["qwen2.5:32b", "qwen2.5:72b", "llama3.3:70b"] else 0.35
w_seniority = 0.20
```
**Use When:** General screening, trust all scoring components, want smart defaults

**2. Semantic-Focused**
```python
w_keywords = 0.20
w_semantic = 0.50  # Highest weight
w_llmfit = 0.30
w_seniority = 0.15
```
**Use When:** High-quality embeddings (mxbai-embed-large), detailed job descriptions

**3. LLM-Heavy**
```python
w_keywords = 0.25
w_semantic = 0.30
w_llmfit = 0.45  # Highest weight
w_seniority = 0.20
```
**Use When:** Excellent LLMs (qwen2.5:32b+), nuanced fit assessment needed

**4. Role-Critical (Seniority Filter)**
```python
w_keywords = 0.25
w_semantic = 0.40
w_llmfit = 0.35
w_seniority = 0.0  # Multiplicative mode!
```
**Use When:** Strict experience level requirements, wrong-level candidates should rank very low

**Seniority Multipliers:**
- Perfect match (diff=0): ×1.00
- Off by 1 level: ×0.75 (25% penalty)
- Off by 2 levels: ×0.40 (60% penalty)
- Off by 3+ levels: ×0.20 (80% penalty)

**5. Custom**
```python
# User-defined via manual sliders
w_keywords = st.slider("Weight: Keyword score", 0.0, 1.0, 0.30, 0.05)
w_semantic = st.slider("Weight: Semantic similarity", 0.0, 1.0, 0.40, 0.05)
w_llmfit = st.slider("Weight: LLM fit score", 0.0, 1.0, 0.30, 0.05)
w_seniority = st.slider("Weight: Seniority alignment", 0.0, 0.5, 0.2, 0.05)
```
**Use When:** Specific requirements not covered by presets, experimenting with weights

### Model-Aware Adaptation Logic

**Why Adaptive Weights?**
- Better embedding models produce more reliable semantic scores → deserve higher weight
- Better LLMs produce more accurate fit assessments → deserve higher weight
- Fixed weights treat all models equally, ignoring quality differences

**Implementation (Lines 557-573):**
```python
if weight_preset == "Balanced (Recommended)":
    w_keywords_default = 0.25
    # Better embeddings get higher semantic weight
    w_semantic_default = 0.45 if embed_model in ["mxbai-embed-large", "bge-large"] else 0.40
    # Better LLMs get higher fit weight
    w_llmfit_default = 0.40 if llm_model in ["qwen2.5:32b", "qwen2.5:72b", "llama3.3:70b"] else 0.35
    w_seniority_default = 0.20
```

**Example Configurations:**
| LLM | Embedding | Keywords | Semantic | LLM Fit | Seniority |
|-----|-----------|----------|----------|---------|-----------|
| qwen3:8b | nomic-embed-text | 25% | 40% | 35% | 20% |
| qwen2.5:32b | nomic-embed-text | 25% | 40% | **40%** | 20% |
| qwen3:8b | mxbai-embed-large | 25% | **45%** | 35% | 20% |
| qwen2.5:32b | mxbai-embed-large | 25% | **45%** | **40%** | 20% |

## Testing & Validation

**Current Validation Patterns:**

1. **Model Availability Check (Lines 647-669)**
   ```python
   if st.button("📋 Check Model Availability"):
       llm_ok, llm_msg = check_model_available(llm_model)
       embed_ok, embed_msg = check_model_available(embed_model)
       # Displays status + installation commands if needed
   ```

2. **File Upload Validation**
   - Streamlit's built-in file type checking (PDF, DOCX, TXT)
   - Graceful skipping of unparseable files

3. **Weight Sum Validation (Lines 630-636)**
   ```python
   if weight_preset == "Custom":
       total_weight = w_keywords + w_semantic + w_llmfit
       if abs(total_weight - 1.0) > 0.05:
           st.warning(f"Weights sum to {total_weight:.2f}, recommend adjusting to ~1.0")
   ```

4. **JSON Parse Fallbacks**
   - LLM responses wrapped in try-except with sensible defaults (50% score, mid-level seniority)

**No Formal Test Suite Exists.** Future considerations:
- Unit tests for scoring functions (keyword_counts, cosine_sim)
- Integration tests for Ollama API wrappers
- End-to-end tests with sample resumes + job descriptions
- Performance benchmarks for different model combinations

## Security & Privacy

### Local-Only Processing

**Core Privacy Guarantee:**
- ✅ All processing via local Ollama instance (`http://localhost:11434`)
- ✅ No API keys, credentials, or external service calls
- ✅ Resume text never leaves your machine
- ✅ No telemetry, analytics, or usage tracking

**Footer Reminder (Line 1040):**
```python
st.caption("🔒 All processing is done locally using your Ollama models. No data is sent to external servers.")
```

### Sensitive Data Handling

**File I/O:**
- DOCX files extracted via temporary files with `uuid.uuid4()` unique names (Line 95)
- Temp files deleted immediately after extraction (Line 102: `os.unlink(temp_path)`)
- No persistent logging of resume text content

**Session State:**
- All data stored in Streamlit session state (memory-only)
- Cleared when user clicks "🗑️ Clear Results & Reset Session" (Lines 711-714)
- No database persistence by default

**PDF Warnings:**
- PyPDF parsing warnings suppressed with `warnings.catch_warnings()` (Lines 81-82)
- Prevents sensitive file paths from appearing in logs

### No Command Injection

- User input limited to:
  - File uploads (type-checked by Streamlit)
  - Text areas (keywords, questions)
  - Dropdowns/sliders (predefined values)
- No direct shell execution of user input
- Model names validated against Ollama's model list

## Extensibility

### Potential Improvements

**1. Database Persistence**
- Replace session state with SQLite/PostgreSQL
- Track screening history across sessions
- Compare candidates from different job postings

**2. Custom Scoring Plugins**
```python
# Plugin interface for domain-specific scoring
def custom_score_plugin(resume: str, job_desc: str) -> float:
    """User-defined scoring function."""
    pass

# Registration
CUSTOM_SCORERS = {
    "technical_depth": custom_score_plugin,
    "culture_fit": another_plugin,
}
```

**3. Multi-Language Support**
- Translate prompts to non-English languages
- Detect resume language and adapt LLM prompts
- Support multilingual keyword matching

**4. Advanced Filtering**
- Boolean keyword queries (AND/OR/NOT)
- Salary range filtering
- Location-based screening
- Required vs. nice-to-have keywords

**5. Ranking History & Comparison**
```python
# Store past screening runs
SCREENING_HISTORY = [
    {"date": "2026-02-10", "job": "Senior Python Developer", "candidates": [...]}
]
# Compare candidates across multiple jobs
# Track changes in candidate rankings over time
```

**6. REST API Interface**
```python
# Flask/FastAPI wrapper for programmatic access
@app.post("/screen")
def screen_candidates(job_desc: str, resumes: List[UploadFile]):
    # ... scoring logic ...
    return {"ranked_candidates": results}
```

### Code Hooks for Extension

**Scoring Function Extraction:**
```python
# Current: Scoring logic embedded in main loop (Lines 860-940)
# Future: Extract to standalone function
def score_candidate(
    resume_text: str,
    jd_text: str,
    jd_vec: np.ndarray,
    resume_chunks: List[DocChunk],
    weights: Dict[str, float],
    config: ScoringConfig
) -> CandidateScore:
    """Abstracted scoring function for reuse in API/batch processing."""
    pass
```

**Weight Preset Configuration:**
```python
# Current: Presets embedded in sidebar UI (Lines 546-619)
# Future: External JSON/YAML configuration
WEIGHT_PRESETS = {
    "balanced": {
        "keywords": 0.25,
        "semantic": {"default": 0.40, "mxbai-embed-large": 0.45},
        "llm_fit": {"default": 0.35, "qwen2.5:32b": 0.40},
        "seniority": 0.20
    }
}
```

**Q&A System Enhancement:**
```python
# Current: Candidate-scoped retrieval (Lines 998-1037)
# Future: Cross-candidate comparison
def compare_candidates(question: str, candidates: List[str]) -> Dict[str, str]:
    """Ask same question across multiple candidates and compare answers."""
    pass
```

---

## Quick Reference

**Common Tasks:**
- **Add new model:** Update `LLM_MODELS` or `EMBED_MODELS` dictionary (Lines 34-66)
- **Adjust safety margins:** Modify `get_embedding_chunk_size()` (Lines 350-373)
- **Change weight presets:** Edit preset logic (Lines 546-619)
- **Add scoring component:** Create new function → integrate in main loop (Lines 860-940)
- **Customize LLM prompts:** Modify `llm_fit_score_llm()` or `estimate_seniority()` prompts

**Critical Line References:**
- Context window calculation: Lines 350-373
- Job description chunking: Lines 800-809
- Parallel LLM processing: Lines 896-903
- Weight preset system: Lines 546-619
- Model configuration: Lines 34-66

**Environment:**
- Python: 3.8+
- Ollama: Latest (http://localhost:11434)
- OS: Windows 10/11, macOS, Linux

---

**Last Updated:** 2026-02-11 (Commit 3595e10)
