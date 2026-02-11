import io
import os
import re
import json
import time
import math
import base64
import queue
import uuid
import string
import pathlib
import tempfile
import requests
import numpy as np
import pandas as pd
import warnings

from typing import List, Dict, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from pypdf import PdfReader
import docx2txt
import faiss

# ----------------------------
# CONFIG
# ----------------------------
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Model presets with descriptions and context windows
LLM_MODELS = {
    "qwen3:8b": {"name": "Qwen 3 8B", "desc": "Fast, good quality (5GB VRAM)", "default": True, "ctx_window": 8192},
    "qwen2.5:14b": {"name": "Qwen 2.5 14B", "desc": "Balanced quality/speed (9GB VRAM)", "default": False, "ctx_window": 32768},
    "qwen2.5:32b": {"name": "Qwen 2.5 32B", "desc": "Excellent quality, recommended (18GB VRAM)", "default": False, "ctx_window": 32768},
    "qwen2.5:72b": {"name": "Qwen 2.5 72B", "desc": "Outstanding quality, slower (40GB+ uses CPU+GPU)", "default": False, "ctx_window": 32768},
    "llama3.3:70b": {"name": "Llama 3.3 70B", "desc": "Alternative high-quality option (40GB+)", "default": False, "ctx_window": 128000},
}

EMBED_MODELS = {
    "nomic-embed-text": {
        "name": "Nomic Embed Text",
        "desc": "Fast, good quality (default)",
        "default": True,
        "ctx_window": 8192
    },
    "mxbai-embed-large": {
        "name": "MxBai Embed Large",
        "desc": "Better semantic understanding",
        "default": False,
        "ctx_window": 512
    },
    "bge-large": {
        "name": "BGE Large",
        "desc": "Alternative high-quality embeddings",
        "default": False,
        "ctx_window": 512
    },
}

# Get defaults from environment or use first default from presets
DEFAULT_LLM = os.environ.get("LLM_MODEL") or next((k for k, v in LLM_MODELS.items() if v["default"]), "qwen3:8b")
DEFAULT_EMBED = os.environ.get("EMBED_MODEL") or next((k for k, v in EMBED_MODELS.items() if v["default"]), "nomic-embed-text")

MAX_CHUNK_TOKENS = 800
CHUNK_OVERLAP = 120

# ----------------------------
# SIMPLE UTILITIES
# ----------------------------
def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\x00", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def read_pdf_bytes(b: bytes) -> str:
    try:
        # Suppress PDF parsing warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            reader = PdfReader(io.BytesIO(b))
            texts = []
            for page in reader.pages:
                texts.append(page.extract_text() or "")
            return clean_text(" ".join(texts))
    except Exception:
        return ""

def read_docx_bytes(b: bytes) -> str:
    # docx2txt expects a path; write to temp
    tmp_dir = tempfile.gettempdir()
    tmp = os.path.join(tmp_dir, f"{uuid.uuid4()}.docx")
    with open(tmp, "wb") as f:
        f.write(b)
    try:
        txt = docx2txt.process(tmp) or ""
        return clean_text(txt)
    except Exception:
        return ""
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

def read_txt_bytes(b: bytes) -> str:
    try:
        return clean_text(b.decode(errors="ignore"))
    except Exception:
        return ""

def split_into_chunks_by_words(text: str, max_words: int = None, overlap: int = CHUNK_OVERLAP):
    # Backward compatibility: if max_words not specified, use old MAX_CHUNK_TOKENS
    if max_words is None:
        max_words = MAX_CHUNK_TOKENS

    words = text.split()
    if not words:
        return []
    step = max(1, max_words - overlap)
    chunks = []
    for start in range(0, len(words), step):
        end = min(len(words), start + max_words)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
    return chunks

# ----------------------------
# OLLAMA API WRAPPERS
# ----------------------------
def check_ollama_connection() -> Tuple[bool, str]:
    """Check if Ollama is running and accessible."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        return True, ""
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to Ollama at {OLLAMA_BASE}. Make sure Ollama is installed and running."
    except requests.exceptions.Timeout:
        return False, f"Connection to Ollama at {OLLAMA_BASE} timed out."
    except Exception as e:
        return False, f"Error connecting to Ollama: {str(e)}"

def check_model_available(model: str) -> Tuple[bool, str]:
    """Check if a specific model is available in Ollama.

    Handles tags like `nomic-embed-text:latest` when the app requests `nomic-embed-text`.
    """
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        models_raw = [m.get("name", "") for m in r.json().get("models", [])]

        # Compare using base names (before ':') so `foo` matches `foo:latest`
        requested_base = model.split(":", 1)[0]
        base_names = [name.split(":", 1)[0] for name in models_raw]

        if requested_base in base_names:
            return True, ""
        else:
            available = ", ".join(models_raw[:5]) if models_raw else "none"
            return False, (
                f"Model '{model}' not found. Available models: {available} "
                f"(run 'ollama pull {model}' to install, or use one of the available tags)"
            )
    except Exception as e:
        return False, f"Error checking models: {str(e)}"

def ollama_embeddings(texts: List[str], model: str = None) -> np.ndarray:
    if model is None:
        model = DEFAULT_EMBED

    # Calculate safe word limit for this model
    max_words = get_embedding_chunk_size(model)

    # batch into multiple calls to avoid huge payloads
    vectors = []
    for idx, t in enumerate(texts):
        # Log progress for debugging slow embeddings
        if len(texts) > 5 and idx % 5 == 0:
            print(f"Embedding chunk {idx+1}/{len(texts)}...")

        # Truncate if needed (safety net)
        words = t.split()
        if len(words) > max_words:
            t = " ".join(words[:max_words])
            print(f"Warning: Truncated embedding input to {max_words} words for {model}")

        payload = {"model": model, "prompt": t}
        try:
            r = requests.post(f"{OLLAMA_BASE}/api/embeddings", json=payload, timeout=60)
            r.raise_for_status()
            response_data = r.json()
            if "embedding" not in response_data:
                raise ValueError(f"No embedding in response: {response_data}")
            vec = np.array(response_data.get("embedding"), dtype=np.float32)
            vectors.append(vec)
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {OLLAMA_BASE}. "
                "Please make sure Ollama is installed and running. "
                "Download from https://ollama.ai"
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"Model '{model}' not found. "
                    f"Please install it with: ollama pull {model}"
                )
            elif e.response.status_code == 500:
                error_msg = "Ollama server error. "
                try:
                    error_detail = e.response.json().get("error", "")
                    if "model" in error_detail.lower() or "not found" in error_detail.lower():
                        error_msg += f"Model '{model}' may not be installed. Run: ollama pull {model}"
                    else:
                        error_msg += f"Details: {error_detail}"
                except:
                    error_msg += "Check Ollama logs for details."
                raise RuntimeError(error_msg)
            else:
                raise RuntimeError(f"Ollama API error (HTTP {e.response.status_code}): {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error getting embeddings: {str(e)}")
    return np.vstack(vectors)

def ollama_chat(messages: List[Dict], model: str = None, json_mode: bool = False, temperature: float=0.2, system_prompt: str=None) -> str:
    if model is None:
        model = DEFAULT_LLM
    body = {
        "model": model,
        "messages": ([] if not system_prompt else [{"role": "system", "content": system_prompt}]) + messages,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    # Many local models ignore tools/json; we instruct format via prompt.
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/chat", json=body, timeout=60)
        r.raise_for_status()
        response_data = r.json()
        if "message" not in response_data or "content" not in response_data["message"]:
            raise ValueError(f"Unexpected response format: {response_data}")
        content = response_data["message"]["content"]
        return content
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE}. "
            "Please make sure Ollama is installed and running."
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise ValueError(
                f"Model '{model}' not found. "
                f"Please install it with: ollama pull {model}"
            )
        elif e.response.status_code == 500:
            error_msg = "Ollama server error. "
            try:
                error_detail = e.response.json().get("error", "")
                if "model" in error_detail.lower() or "not found" in error_detail.lower():
                    error_msg += f"Model '{model}' may not be installed. Run: ollama pull {model}"
                else:
                    error_msg += f"Details: {error_detail}"
            except:
                error_msg += "Check Ollama logs for details."
            raise RuntimeError(error_msg)
        else:
            raise RuntimeError(f"Ollama API error (HTTP {e.response.status_code}): {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error in chat API: {str(e)}")

# ----------------------------
# SCORING
# ----------------------------
@dataclass
class CandidateScore:
    name: str
    file_name: str
    keyword_score: float
    semantic_score: float
    llm_fit_score: float
    final_score: float
    top_keywords: List[Tuple[str, int]]
    summary: str
    seniority_score: float = 0.0
    seniority_rationale: str = ""

def keyword_counts(text: str, keywords: Dict[str, float]) -> Tuple[float, List[Tuple[str,int]]]:
    text_lc = text.lower()
    # normalize punctuation to spaces
    text_lc = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text_lc)
    tokens = text_lc.split()
    counts = {}
    weighted_total = 0.0
    
    for kw, w in keywords.items():
        # allow quoted phrases; else bag-of-words
        k = kw.strip().lower()
        if " " in k:
            # phrase count
            count = len(re.findall(r"\b" + re.escape(k) + r"\b", text_lc))
        else:
            count = sum(1 for t in tokens if t == k)
        
        counts[k] = count
        weighted_total += count * float(w)
    
    # Normalize by total weighted occurrences + text length heuristic
    # Keep simple: sqrt to dampen very long resumes
    norm = weighted_total / max(1.0, math.sqrt(len(tokens)/250.0))
    top = sorted([(k, counts[k]) for k in counts], key=lambda x: x[1], reverse=True)[:10]
    return norm, top

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def get_context_limits(model: str) -> Tuple[int, int]:
    """Calculate safe context limits for JD and resume based on model's context window.

    Returns: (max_jd_chars, max_resume_chars)
    """
    # Get model's context window from config
    ctx_window = LLM_MODELS.get(model, {}).get("ctx_window", 8192)

    # Calculate safe character limits
    # Reserve space for: prompt instructions (~500 tokens), response (~200 tokens), safety margin (30%)
    # Approximate: 1 token ≈ 4 characters
    usable_tokens = int(ctx_window * 0.7)  # 30% safety margin
    usable_chars = usable_tokens * 4

    # Allocate 40% to JD, 60% to resume
    max_jd_chars = int(usable_chars * 0.4)
    max_resume_chars = int(usable_chars * 0.6)

    # Set minimums for very small models
    max_jd_chars = max(1500, max_jd_chars)
    max_resume_chars = max(2000, max_resume_chars)

    return max_jd_chars, max_resume_chars

def get_embedding_chunk_size(embed_model: str) -> int:
    """Calculate safe chunk size (in words) for the embedding model.

    Returns: max_chunk_words
    """
    # Get embedding model's context window
    ctx_window = EMBED_MODELS.get(embed_model, {}).get("ctx_window", 8192)

    # Apply safety margin (30%) and convert tokens to words
    # More accurate ratio: 1 token ≈ 0.75 words (or 1 word ≈ 1.33 tokens)
    safe_tokens = int(ctx_window * 0.7)  # 30% safety margin
    safe_words = int(safe_tokens * 0.75)  # Convert to words

    # Set reasonable minimum
    safe_words = max(200, safe_words)

    return safe_words

def llm_fit_score_llm(resume_text: str, job_desc: str, model: str = None) -> Tuple[float, str]:
    """Ask the LLM for a structured 0–100 fit score with a short rationale."""
    if model is None:
        model = DEFAULT_LLM

    # Calculate context limits based on model's capacity
    max_jd_chars, max_resume_chars = get_context_limits(model)

    prompt = f"""
You are scoring resume-job fit.

Return ONLY a JSON object with fields:
- "score": an integer 0-100 (no decimals)
- "rationale": one short sentence (max 30 words)

Criteria:
- Required skills/experience match
- Years/seniority alignment
- Domain alignment (e.g., IEQ/HAZ/Insurance if present)
- Red flags or gaps

JOB DESCRIPTION:
\"\"\"{job_desc[:max_jd_chars]}\"\"\"

RESUME:
\"\"\"{resume_text[:max_resume_chars]}\"\"\"
"""
    resp = ollama_chat([{"role":"user", "content": prompt}], model=model, temperature=0.1)
    # Try to parse minimal JSON in messy outputs
    try:
        start = resp.find("{"); end = resp.rfind("}")
        obj = json.loads(resp[start:end+1])
        score = int(obj.get("score", 0))
        rationale = str(obj.get("rationale", "")).strip()
        score = max(0, min(100, score))
        return float(score), rationale
    except Exception:
        # fallback simple heuristic
        return 50.0, "Heuristic fallback due to parse error."

def estimate_seniority(resume_text: str, model: str = None) -> Tuple[int, str]:
    """Estimate candidate seniority level using LLM analysis."""
    if model is None:
        model = DEFAULT_LLM

    # Calculate context limits based on model's capacity
    _, max_resume_chars = get_context_limits(model)

    prompt = f"""
Analyze this resume and determine the candidate's seniority level.

Return ONLY a JSON object with fields:
- "seniority": an integer 1-4 (no decimals)
- "rationale": one short sentence explaining the assessment (max 30 words)

Seniority Scale:
1 = Junior (0-3 years experience, entry-level, recent graduate)
2 = Experienced (3-8 years experience, mid-level, some specialization)
3 = Senior (8-15 years experience, senior-level, deep expertise)
4 = Leadership (15+ years experience, management/executive roles, team leadership)

Consider: years of experience, job titles, management responsibilities, project complexity, team size led.

RESUME:
\"\"\"{resume_text[:max_resume_chars]}\"\"\"
"""
    resp = ollama_chat([{"role":"user", "content": prompt}], model=model, temperature=0.1)
    # Try to parse minimal JSON in messy outputs
    try:
        start = resp.find("{"); end = resp.rfind("}")
        obj = json.loads(resp[start:end+1])
        seniority = int(obj.get("seniority", 2))
        rationale = str(obj.get("rationale", "")).strip()
        seniority = max(1, min(4, seniority))  # Clamp to valid range
        return seniority, rationale
    except Exception:
        # fallback to experienced level
        return 2, "Heuristic fallback due to parse error."

# ----------------------------
# INDEXING & RETRIEVAL
# ----------------------------
@dataclass
class DocChunk:
    candidate: str
    file_name: str
    chunk_id: str
    text: str
    vector: np.ndarray

class ResumeIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.ids: List[str] = []
        self.meta: Dict[str, DocChunk] = {}

    def add(self, chunks: List[DocChunk]):
        if not chunks:
            return
        vecs = np.vstack([c.vector for c in chunks]).astype(np.float32)
        faiss.normalize_L2(vecs)  # cosine via inner-product on normalized vectors
        self.index.add(vecs)
        for c in chunks:
            self.ids.append(c.chunk_id)
            self.meta[c.chunk_id] = c

    def search(self, query_vec: np.ndarray, k: int = 5):
        q = query_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        results = []
        for i, score in zip(I[0], D[0]):
            if i < 0 or i >= len(self.ids): 
                continue
            cid = self.ids[i]
            meta = self.meta[cid]
            results.append((score, meta))
        return results

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Local Resume Screener (Qwen + FAISS)", layout="wide")

st.title("🔎 Local Resume Screener (Privacy-First)")
st.caption("Loads resumes, compares to Job Description + Keywords, ranks candidates, and supports Q&A over selected profiles. Runs on your local LLM via Ollama.")

with st.sidebar:
    st.header("⚙️ Model Configuration")

    # LLM Model Selection
    llm_options = [f"{k} - {v['desc']}" for k, v in LLM_MODELS.items()]
    llm_keys = list(LLM_MODELS.keys())
    default_llm_idx = llm_keys.index(DEFAULT_LLM) if DEFAULT_LLM in llm_keys else 0

    llm_selection = st.selectbox(
        "Language Model",
        llm_options,
        index=default_llm_idx,
        help="Choose based on your GPU VRAM and quality needs"
    )
    llm_model = llm_keys[llm_options.index(llm_selection)]

    # Embedding Model Selection
    embed_options = [f"{k} - {v['desc']}" for k, v in EMBED_MODELS.items()]
    embed_keys = list(EMBED_MODELS.keys())
    default_embed_idx = embed_keys.index(DEFAULT_EMBED) if DEFAULT_EMBED in embed_keys else 0

    embed_selection = st.selectbox(
        "Embedding Model",
        embed_options,
        index=default_embed_idx,
        help="Model used for semantic similarity"
    )
    embed_model = embed_keys[embed_options.index(embed_selection)]

    # Advanced: Allow custom model input
    with st.expander("🔧 Advanced: Custom Models"):
        use_custom_llm = st.checkbox("Use custom LLM model")
        if use_custom_llm:
            llm_model = st.text_input("Custom LLM", llm_model)

        use_custom_embed = st.checkbox("Use custom embedding model")
        if use_custom_embed:
            embed_model = st.text_input("Custom Embedding", embed_model)

    st.divider()
    st.header("⚖️ Scoring Weights")

    # Weight preset selector
    weight_preset = st.selectbox(
        "Weight Preset",
        [
            "Balanced (Recommended)",
            "Semantic-Focused",
            "LLM-Heavy",
            "Role-Critical (Seniority Filter)",
            "Custom"
        ],
        help="Choose a pre-configured weight distribution optimized for your model selection, or customize your own"
    )

    # Apply preset weights based on selection
    if weight_preset == "Balanced (Recommended)":
        # Adaptive based on model quality
        w_keywords_default = 0.25
        # Better embeddings get higher semantic weight
        w_semantic_default = 0.45 if embed_model in ["mxbai-embed-large", "bge-large"] else 0.40
        # Better LLMs get higher fit weight
        w_llmfit_default = 0.40 if llm_model in ["qwen2.5:32b", "qwen2.5:72b", "llama3.3:70b"] else 0.35
        w_seniority_default = 0.20

    elif weight_preset == "Semantic-Focused":
        # Prioritize embedding-based similarity
        w_keywords_default = 0.20
        w_semantic_default = 0.50
        w_llmfit_default = 0.30
        w_seniority_default = 0.15

    elif weight_preset == "LLM-Heavy":
        # Prioritize LLM reasoning and fit assessment
        w_keywords_default = 0.25
        w_semantic_default = 0.30
        w_llmfit_default = 0.45
        w_seniority_default = 0.20

    elif weight_preset == "Role-Critical (Seniority Filter)":
        # Use seniority as multiplicative filter
        w_keywords_default = 0.25
        w_semantic_default = 0.40
        w_llmfit_default = 0.35
        w_seniority_default = 0.0  # Multiplicative mode

    else:  # Custom
        # Use legacy defaults for manual adjustment
        w_keywords_default = 0.30
        w_semantic_default = 0.40
        w_llmfit_default = 0.30
        w_seniority_default = 0.20

    # Show weights based on preset selection
    if weight_preset == "Custom":
        # Show manual sliders for custom configuration
        w_keywords = st.slider("Weight: Keyword score", 0.0, 1.0, w_keywords_default, 0.05)
        w_semantic = st.slider("Weight: Semantic similarity", 0.0, 1.0, w_semantic_default, 0.05)
        w_llmfit   = st.slider("Weight: LLM fit score", 0.0, 1.0, w_llmfit_default, 0.05)
        w_seniority = st.slider("Weight: Seniority alignment", 0.0, 0.5, w_seniority_default, 0.05)
    else:
        # Show preset weights as read-only info
        seniority_mode = "Multiplicative Filter" if w_seniority_default == 0 else "Additive Bonus"
        st.info(f"""
**Active Weights:**
- **Keywords:** {w_keywords_default:.0%} (exact keyword matching)
- **Semantic:** {w_semantic_default:.0%} (embedding-based similarity)
- **LLM Fit:** {w_llmfit_default:.0%} (holistic fit assessment)
- **Seniority:** {w_seniority_default:.0%} ({seniority_mode})

*Preset weights are optimized for your selected models: {llm_model} + {embed_model}*
        """)
        # Apply preset values
        w_keywords = w_keywords_default
        w_semantic = w_semantic_default
        w_llmfit = w_llmfit_default
        w_seniority = w_seniority_default

    # Performance optimization option
    skip_seniority = st.checkbox(
        "⚡ Skip seniority scoring (2x faster)",
        value=False,
        help="Skip seniority assessment to reduce processing time by ~50%. Uses expected role level as default."
    )
    
    # Weight validation and explanation
    if weight_preset == "Custom":
        # Provide validation for custom weights
        if w_seniority > 0:
            total_weight = w_keywords + w_semantic + w_llmfit + w_seniority
            st.info(f"Total weight: {total_weight:.2f} (additive seniority scoring)")
        else:
            total_weight = w_keywords + w_semantic + w_llmfit
            if abs(total_weight - 1.0) > 0.05:
                st.warning(f"Weights sum to {total_weight:.2f}, recommend adjusting to ~1.0")
    else:
        # Explain preset mode
        if w_seniority > 0:
            st.caption(f"ℹ️ Seniority adds +{w_seniority:.0%} bonus/penalty (candidates can exceed 100% score)")
        else:
            st.caption("ℹ️ Seniority acts as filter (multiplies final score based on role match)")

    # Model availability check
    st.divider()
    if st.button("📋 Check Model Availability"):
        with st.spinner("Checking Ollama models..."):
            conn_ok, conn_msg = check_ollama_connection()
            if not conn_ok:
                st.error(conn_msg)
            else:
                llm_ok, llm_msg = check_model_available(llm_model)
                embed_ok, embed_msg = check_model_available(embed_model)

                if llm_ok and embed_ok:
                    st.success("✅ Both models are installed and ready!")
                else:
                    if not llm_ok:
                        st.error(f"❌ LLM: {llm_msg}")
                        st.code(f"ollama pull {llm_model}", language="bash")
                    else:
                        st.success(f"✅ LLM model '{llm_model}' is available")

                    if not embed_ok:
                        st.error(f"❌ Embedding: {embed_msg}")
                        st.code(f"ollama pull {embed_model}", language="bash")
                    else:
                        st.success(f"✅ Embedding model '{embed_model}' is available")

st.subheader("1) Upload Inputs")

jd_file = st.file_uploader("Job Description (TXT / PDF / DOCX)", type=["txt","pdf","docx"])

# Seniority Level Selection with integrated mapping
seniority_options = {
    "Junior (0-3 years)": 1,
    "Experienced (3-8 years)": 2,
    "Senior (8-15 years)": 3,
    "Leadership (15+ years)": 4
}

seniority_level = st.selectbox(
    "Expected Role Level",
    list(seniority_options.keys()),
    help="Select the expected seniority level for this position"
)

required_level = seniority_options[seniority_level]

# Keywords section with two separate boxes
st.subheader("Keywords")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Must Haves** (Weight: 5)")
    must_have_text = st.text_area("Required skills/experience", height=120, placeholder="asbestos\nIEQ\nLegionella\nhazardous materials\ninsurance\nbilingual\n...", key="must_haves")

with col2:
    st.markdown("**Nice to Haves** (Weight: 3)")
    nice_to_have_text = st.text_area("Preferred skills/experience", height=120, placeholder="project management\ncertifications\nleadership\n...", key="nice_to_haves")

resume_files = st.file_uploader("Resumes (multiple; PDF / DOCX / TXT)", accept_multiple_files=True, type=["pdf","docx","txt"])

colA, colB = st.columns([1,1])
with colA:
    start_button = st.button("Build Index & Rank Candidates", type="primary")
with colB:
    clear_button = st.button("Clear session results")

if clear_button:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.success("Cleared session state. Ready for fresh run.")

def parse_keywords(must_have_text: str, nice_to_have_text: str) -> Dict[str,float]:
    """Parse keywords from must-haves and nice-to-haves text areas with fixed weights."""
    out = {}
    
    # Parse must-haves with weight 5
    for line in must_have_text.splitlines():
        line = line.strip()
        if line:
            out[line] = 5.0
    
    # Parse nice-to-haves with weight 3
    for line in nice_to_have_text.splitlines():
        line = line.strip()
        if line:
            out[line] = 3.0
    
    return out

def extract_text_from_upload(f) -> str:
    name = f.name.lower()
    data = f.read()
    if name.endswith(".pdf"):
        return read_pdf_bytes(data)
    elif name.endswith(".docx"):
        return read_docx_bytes(data)
    else:
        return read_txt_bytes(data)

if start_button:
    if not jd_file or not resume_files:
        st.error("Please upload a Job Description and at least one resume.")
    else:
        # Validate Ollama connection and models before processing
        with st.spinner("Checking Ollama connection…"):
            conn_ok, conn_msg = check_ollama_connection()
            if not conn_ok:
                st.error(conn_msg)
                st.stop()
            
            embed_ok, embed_msg = check_model_available(embed_model)
            if not embed_ok:
                st.error(f"Embedding model issue: {embed_msg}")
                st.stop()
            
            llm_ok, llm_msg = check_model_available(llm_model)
            if not llm_ok:
                st.error(f"LLM model issue: {llm_msg}")
                st.stop()
        
        with st.spinner("Reading and embedding…"):
            try:
                # Job Description
                JD = extract_text_from_upload(jd_file)
                if not JD or len(JD.strip()) < 10:
                    st.error("Failed to extract text from job description. Please try a different file format.")
                    st.stop()

                # Keywords
                keywords = parse_keywords(must_have_text or "", nice_to_have_text or "")

                # Ingest resumes
                candidates_raw = []  # (name, file_name, full_text)
                for rf in resume_files:
                    try:
                        txt = extract_text_from_upload(rf)
                        if txt and len(txt.strip()) > 10:
                            # Guess candidate name from file name (fallback)
                            base = pathlib.Path(rf.name).stem
                            candidate_name = re.sub(r"[_-]+", " ", base).strip()
                            candidates_raw.append((candidate_name, rf.name, txt))
                        else:
                            st.warning(f"Could not extract meaningful text from {rf.name}. File may be corrupted or empty.")
                    except Exception as e:
                        st.warning(f"Failed to read {rf.name}: {e}")
                
                if not candidates_raw:
                    st.error("No valid resumes were processed. Please check your files and try again.")
                    st.stop()

                # Embedding dimension probe
                probe_vec = ollama_embeddings(["probe"], model=embed_model)[0]
                dim = probe_vec.shape[0]
                index = ResumeIndex(dim=dim)

                # Precompute JD embedding
                jd_vec = ollama_embeddings([JD], model=embed_model)[0]
            except (ConnectionError, ValueError, RuntimeError) as e:
                st.error(f"Error during processing: {str(e)}")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                st.stop()

            # Build chunk-level store
            all_chunks: List[DocChunk] = []
            candidate_meta = {}

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_candidates = len(candidates_raw)

            for idx, (cand_name, file_name, fulltext) in enumerate(candidates_raw):
                status_text.text(f"Processing {idx+1}/{total_candidates}: {cand_name}...")
                progress_bar.progress((idx) / total_candidates)

                fulltext = clean_text(fulltext)
                # Calculate safe chunk size for selected embedding model
                max_chunk_words = get_embedding_chunk_size(embed_model)
                chunks = split_into_chunks_by_words(fulltext, max_words=max_chunk_words)
                if not chunks:
                    continue
                vecs = ollama_embeddings(chunks, model=embed_model)
                # Assemble DocChunk objects
                ch_objs = []
                for i, (t, v) in enumerate(zip(chunks, vecs)):
                    ch = DocChunk(
                        candidate=cand_name,
                        file_name=file_name,
                        chunk_id=str(uuid.uuid4()),
                        text=t,
                        vector=v.astype(np.float32),
                    )
                    ch_objs.append(ch)
                index.add(ch_objs)
                all_chunks.extend(ch_objs)
                candidate_meta[cand_name] = {
                    "file_name": file_name,
                    "fulltext": fulltext,
                    "chunks": ch_objs
                }

            progress_bar.progress(1.0)
            status_text.text("Embedding complete. Starting candidate scoring...")

            # Score each candidate
            ranked: List[CandidateScore] = []
            total_for_scoring = len(candidate_meta)
            for score_idx, (cand_name, meta) in enumerate(candidate_meta.items()):
                status_text.text(f"Scoring {score_idx+1}/{total_for_scoring}: {cand_name}...")
                progress_bar.progress(score_idx / total_for_scoring)
                try:
                    fulltext = meta["fulltext"]

                    # (1) keyword score
                    kw_score_raw, top_kw = keyword_counts(fulltext, keywords) if keywords else (0.0, [])

                    # Normalize keyword to 0..100 (logistic-ish)
                    kw_score = 100.0 * (1.0 - math.exp(-0.1 * kw_score_raw))
                    kw_score = max(0.0, min(100.0, kw_score))

                    # (2) semantic similarity (best N chunks vs JD)
                    if meta["chunks"]:
                        chunk_vecs = np.vstack([c.vector for c in meta["chunks"]]).astype(np.float32)
                        # cosine with JD
                        # normalize for cosine
                        q = jd_vec.reshape(1, -1).astype(np.float32)
                        faiss.normalize_L2(q)
                        faiss.normalize_L2(chunk_vecs)
                        sims = (chunk_vecs @ q.T).flatten()
                        topk = float(np.mean(sorted(sims, reverse=True)[:5])) if len(sims) >= 5 else float(np.mean(sims))
                        sem_score = max(0.0, min(100.0, (topk + 1) * 50.0))  # map [-1,1]=>[0,100]
                    else:
                        sem_score = 0.0

                    # (3) & (4) LLM scoring - parallelize if both needed
                    if skip_seniority:
                        # Fast path: only run fit scoring
                        fit_score, rationale = llm_fit_score_llm(fulltext, JD, llm_model)
                        cand_level = required_level  # Default to expected level
                        seniority_rationale = "Skipped for performance"
                    else:
                        # Full scoring: parallelize both LLM calls
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            fit_future = executor.submit(llm_fit_score_llm, fulltext, JD, llm_model)
                            seniority_future = executor.submit(estimate_seniority, fulltext, llm_model)

                            # Wait for both to complete
                            fit_score, rationale = fit_future.result()
                            cand_level, seniority_rationale = seniority_future.result()
                    diff = abs(required_level - cand_level)
                    
                    # Map difference to seniority match score
                    if diff == 0:
                        seniority_score = 100.0
                    elif diff == 1:
                        seniority_score = 75.0
                    elif diff == 2:
                        seniority_score = 40.0
                    else:  # diff >= 3
                        seniority_score = 20.0

                    # Calculate final score based on seniority weight mode
                    if w_seniority > 0:
                        # Additive mode: add seniority as additional component
                        final = w_keywords * kw_score + w_semantic * sem_score + w_llmfit * fit_score + w_seniority * seniority_score
                    else:
                        # Multiplicative mode: apply seniority as multiplier
                        final = (w_keywords * kw_score + w_semantic * sem_score + w_llmfit * fit_score) * (seniority_score / 100.0)

                    ranked.append(CandidateScore(
                        name=cand_name,
                        file_name=meta["file_name"],
                        keyword_score=round(kw_score,2),
                        semantic_score=round(sem_score,2),
                        llm_fit_score=round(fit_score,2),
                        final_score=round(final,2),
                        top_keywords=top_kw,
                        summary=rationale,
                        seniority_score=round(seniority_score,2),
                        seniority_rationale=seniority_rationale
                    ))
                except (ConnectionError, ValueError, RuntimeError) as e:
                    st.warning(f"Failed to score candidate '{cand_name}': {str(e)}")
                    continue
                except Exception as e:
                    st.warning(f"Unexpected error scoring candidate '{cand_name}': {str(e)}")
                    continue

            ranked = sorted(ranked, key=lambda x: x.final_score, reverse=True)

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Save to session
            st.session_state["JD"] = JD
            st.session_state["keywords"] = keywords
            st.session_state["index"] = index
            st.session_state["candidate_meta"] = candidate_meta
            st.session_state["ranked"] = ranked
            st.session_state["llm_model"] = llm_model
            st.session_state["embed_model"] = embed_model

        st.success("Index built and candidates ranked.")

# ----------------------------
# RESULTS
# ----------------------------
if "ranked" in st.session_state:
    st.subheader("2) Ranked Candidates")

    df = pd.DataFrame([{
        "Candidate": r.name,
        "File": r.file_name,
        "Final Score": r.final_score,
        "Keyword": r.keyword_score,
        "Semantic": r.semantic_score,
        "LLM Fit": r.llm_fit_score,
        "Seniority Match": r.seniority_score,
        "Fit Summary": r.summary,
        "Seniority Notes": r.seniority_rationale
    } for r in st.session_state["ranked"]])

    st.dataframe(df, width='stretch')

    # Download CSV
    csv = df.to_csv(index=False).encode()
    st.download_button("Download Results (CSV)", csv, "ranked_candidates.csv", "text/csv")

    st.divider()
    st.subheader("3) Q&A on a Selected Candidate")

    names = [r.name for r in st.session_state["ranked"]]
    pick = st.selectbox("Choose candidate", names)
    question = st.text_input("Ask a question (e.g., 'Summarize IEQ experience and certifications')", "")
    k_ctx = st.slider("Context chunks", 3, 12, 6)

    if st.button("Answer from Resume", disabled=not question.strip()):
        meta = st.session_state["candidate_meta"][pick]
        index: ResumeIndex = st.session_state["index"]
        embed_model = st.session_state["embed_model"]
        llm_model = st.session_state["llm_model"]

        try:
            with st.spinner("Retrieving and answering…"):
                # Build candidate-only mini index: filter chunks by candidate
                cand_chunks = meta["chunks"]
                if not cand_chunks:
                    st.error("No chunks available for this candidate.")
                    st.stop()
                
                # For retrieval simplicity, embed question and score against candidate chunks locally
                q_vec = ollama_embeddings([question], model=embed_model)[0]
                # Manual search over candidate chunks (no cross-candidate bleed)
                C = np.vstack([c.vector for c in cand_chunks]).astype(np.float32)
                faiss.normalize_L2(C)
                qn = q_vec.reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(qn)
                sims = (C @ qn.T).flatten()
                top_idx = np.argsort(-sims)[:k_ctx]
                contexts = [cand_chunks[i].text for i in top_idx]

                sys = "You strictly answer using the provided CONTEXT. If information is absent, say you cannot find it in the resume."
                content = f"CONTEXT:\n\n" + "\n\n---\n\n".join(contexts[:k_ctx]) + f"\n\nQUESTION:\n{question}\n\nAnswer concisely and cite excerpts when relevant."

                answer = ollama_chat(
                    [{"role":"user","content": content}],
                    model=llm_model,
                    temperature=0.1,
                    system_prompt=sys
                )

            st.markdown("**Answer:**")
            st.write(answer)

            with st.expander("Show retrieved excerpts"):
                for i, (idx, sim) in enumerate(zip(top_idx, sims[top_idx])):
                    st.markdown(f"**Excerpt {i+1}** (sim={float(sim):.3f})")
                    st.write(cand_chunks[idx].text[:2000])
        except (ConnectionError, ValueError, RuntimeError) as e:
            st.error(f"Error during Q&A: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

# Footer note
st.caption("All processing occurs locally. No data leaves your machine unless you change the code.")
