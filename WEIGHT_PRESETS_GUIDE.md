# Weight Preset Guide

The Resume Screening Tool now features intelligent weight presets that automatically optimize scoring based on your selected models.

## Understanding Scoring Components

Your final candidate score combines four weighted factors:

| Component | What It Measures | Model Dependency |
|-----------|------------------|------------------|
| **Keywords** | Exact keyword frequency in resume | None (simple text matching) |
| **Semantic** | Resume-job description similarity | **High** (embedding model quality matters) |
| **LLM Fit** | Holistic assessment of candidate fit | **High** (LLM model quality matters) |
| **Seniority** | Career level alignment | **Moderate** (LLM-based, but coarse scoring) |

## Available Presets

### 1. Balanced (Recommended) ⭐

**Best for:** General use with any model combination

**Weights:**
- Keywords: 25%
- Semantic: 40-45% (adapts based on embedding model)
- LLM Fit: 35-40% (adapts based on LLM model)
- Seniority: 20% (additive bonus/penalty)

**Model-Aware Adjustments:**
- Using `mxbai-embed-large` or `bge-large`? Semantic → 45%
- Using `qwen2.5:32b`, `qwen2.5:72b`, or `llama3.3:70b`? LLM Fit → 40%

**When to use:**
- First time screening resumes
- You want smart defaults without manual tuning
- You trust both semantic matching and LLM reasoning

**Example:** With `qwen2.5:32b` + `mxbai-embed-large`, you get:
```
Keywords:  25%
Semantic:  45% ← Higher quality embeddings
LLM Fit:   40% ← Better LLM reasoning
Seniority: 20%
Total:    130% (seniority adds bonus/penalty)
```

---

### 2. Semantic-Focused

**Best for:** High-quality embedding models, semantic understanding priority

**Weights:**
- Keywords: 20%
- Semantic: 50% ← Highest weight
- LLM Fit: 30%
- Seniority: 15% (additive)

**When to use:**
- You've installed `mxbai-embed-large` or `bge-large`
- Your job description is detailed and well-written
- You want to prioritize overall skill/experience similarity
- Semantic matching is more reliable than LLM reasoning for your use case

**Trade-off:** Less emphasis on LLM's holistic "fit" assessment

---

### 3. LLM-Heavy

**Best for:** Excellent LLM models, reasoning-based assessment priority

**Weights:**
- Keywords: 25%
- Semantic: 30%
- LLM Fit: 45% ← Highest weight
- Seniority: 20% (additive)

**When to use:**
- You're using `qwen2.5:32b`, `qwen2.5:72b`, or `llama3.3:70b`
- Your job requires nuanced fit assessment beyond skills
- You value LLM's reasoning about culture fit, growth potential, etc.
- You're willing to wait longer for better LLM analysis

**Trade-off:** Processing takes longer due to heavy LLM usage

---

### 4. Role-Critical (Seniority Filter)

**Best for:** When role level match is critical (e.g., must be Senior+)

**Weights:**
- Keywords: 25%
- Semantic: 40%
- LLM Fit: 35%
- Seniority: 0% ← **Multiplicative mode**

**How it works:**
Unlike other presets, seniority doesn't add to the score—it **multiplies** the final score:

| Seniority Match | Score Multiplier | Effect |
|----------------|------------------|---------|
| Perfect match (diff = 0) | ×1.00 | No change |
| Off by 1 level | ×0.75 | 25% penalty |
| Off by 2 levels | ×0.40 | 60% penalty |
| Off by 3+ levels | ×0.20 | 80% penalty |

**When to use:**
- Junior candidates are completely unsuitable for a Senior role
- Leadership role requires actual management experience
- You want wrong-level candidates to rank very low
- Role level mismatch is a dealbreaker

**Example:** A candidate with 85% base score but 2 levels too junior:
```
Base score: 85%
Seniority multiplier: ×0.40 (2 levels off)
Final score: 85% × 0.40 = 34% ← Heavily penalized
```

---

### 5. Custom

**Best for:** Advanced users who want full manual control

**What you get:**
- Manual sliders for all four weights
- No automatic adjustments
- Full flexibility to experiment

**When to use:**
- You have specific requirements not covered by presets
- You're experimenting to find optimal weights for your hiring process
- You understand the trade-offs between different components

---

## How Seniority Modes Work

### Additive Mode (Presets 1-3)
When seniority weight > 0, it acts as a **bonus/penalty** added to the score:

```
Final Score = (Keywords × 25%) + (Semantic × 40%) + (LLM Fit × 35%) + (Seniority × 20%)
```

- **Perfect match:** +20% to final score
- **Close match:** +10-15% to final score
- **Poor match:** +0-5% to final score
- **Total can exceed 100%** (e.g., 115% for excellent candidate with perfect level match)

### Multiplicative Mode (Preset 4: Role-Critical)
When seniority weight = 0, it acts as a **dampening filter**:

```
Base Score = (Keywords × 25%) + (Semantic × 40%) + (LLM Fit × 35%)
Final Score = Base Score × Seniority_Multiplier
```

- **Heavily penalizes** wrong-level candidates
- **Total stays at or below 100%**
- Acts like a role-level gatekeeper

---

## Choosing the Right Preset

### Decision Tree

```
Do you need strict role-level filtering?
├─ YES → Use "Role-Critical (Seniority Filter)"
└─ NO  → Continue below

Is your embedding model high-quality (mxbai-embed-large, bge-large)?
├─ YES → Consider "Semantic-Focused"
└─ NO  → Continue below

Is your LLM model excellent (qwen2.5:32b+, llama3.3:70b)?
├─ YES → Consider "LLM-Heavy"
└─ NO  → Continue below

Want smart defaults without manual tuning?
├─ YES → Use "Balanced (Recommended)" ⭐
└─ NO  → Use "Custom" and experiment
```

### Quick Reference

| Your Models | Recommended Preset |
|-------------|-------------------|
| `qwen3:8b` + `nomic-embed-text` | Balanced |
| `qwen2.5:32b` + `nomic-embed-text` | LLM-Heavy |
| `qwen3:8b` + `mxbai-embed-large` | Semantic-Focused |
| `qwen2.5:32b` + `mxbai-embed-large` | Balanced (auto-adjusts to 45%/40%) |
| Any models, strict level match needed | Role-Critical |

---

## Tips for Using Presets

### 1. Start with Balanced
The "Balanced (Recommended)" preset automatically adapts to your model selection and provides good results for most use cases.

### 2. Try Different Presets on Same Batch
Process the same 5-10 resumes with different presets to see how rankings change. This helps you understand what each preset emphasizes.

### 3. Use Role-Critical Sparingly
Multiplicative seniority is powerful but harsh. Reserve it for cases where level mismatch is truly a dealbreaker.

### 4. Upgrade Models Before Changing Presets
If you're not satisfied with results, consider upgrading to better models first:
- `qwen3:8b` → `qwen2.5:32b` (4x better reasoning)
- `nomic-embed-text` → `mxbai-embed-large` (2x better semantics)

Then let the Balanced preset auto-adjust weights for you.

### 5. Custom Mode for Experimentation
Use Custom mode to test hypotheses like:
- "What if I ignore keywords completely?" (set to 0%)
- "What if semantic gets 70%?" (see if it over-prioritizes similar but unqualified candidates)

---

## Understanding the Active Weights Display

When you select a preset, the app shows:

```
Active Weights:
- Keywords: 25% (exact keyword matching)
- Semantic: 45% (embedding-based similarity)
- LLM Fit: 40% (holistic fit assessment)
- Seniority: 20% (Additive Bonus)

Preset weights are optimized for your selected models: qwen2.5:32b + mxbai-embed-large
```

This tells you:
1. **What weights are active** for each component
2. **What each component does** (quick reminder)
3. **Seniority mode** (Additive Bonus vs Multiplicative Filter)
4. **Which models** triggered the automatic adjustments

---

## FAQ

### Q: Can I see the sliders even in preset mode?
**A:** No, preset modes show read-only weights. Select "Custom" to see and adjust sliders manually.

### Q: Do presets change when I switch models?
**A:** Yes! "Balanced" preset watches your model selection and auto-adjusts. If you switch from `qwen3:8b` to `qwen2.5:32b`, LLM Fit weight increases from 35% to 40%.

### Q: Why does Balanced give semantic 40% and LLM fit only 35% by default?
**A:** Embedding-based semantic matching is generally more reliable and consistent than smaller LLM models. When you upgrade to `qwen2.5:32b` or better, LLM Fit increases to 40% because those models are excellent at holistic reasoning.

### Q: What if I want 60% semantic and 40% LLM fit?
**A:** Select "Custom" preset and adjust sliders to your preference.

### Q: When should keywords get more than 25%?
**A:** Rarely. Keyword matching is limited because it doesn't understand semantics. If your job has critical must-have technologies (e.g., "must know Kubernetes"), use the "Must-Have Keywords" feature in the main interface instead of increasing keyword weight.

### Q: Can total weight exceed 100%?
**A:** Yes, when using additive seniority (presets 1-3). A candidate can score 115% if they perfectly match the role level. This is intentional—it rewards ideal-level candidates.

---

## Advanced: Why These Weights?

Based on analysis of component reliability:

| Component | Reliability Variance | Rationale for Weight |
|-----------|---------------------|---------------------|
| **Keywords** | Low (deterministic) | 20-25%: Simple matching, no semantic understanding |
| **Semantic** | 2x variance (model-dependent) | 40-50%: Most reliable for overall fit when using quality embeddings |
| **LLM Fit** | 3-4x variance (model-dependent) | 30-45%: Excellent with good models, but slower and costlier |
| **Seniority** | Moderate (coarse 4-level scale) | 15-20%: Useful signal but limited granularity |

**Higher quality models = higher weights** because their outputs are more trustworthy.
