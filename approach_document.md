# SHL Assessment Recommender — Approach Document

**Candidate Submission | AI Intern Role | SHL Labs**

---

## Problem & Design Philosophy

Hiring managers often can't express what they need in catalog keywords. The goal is a **conversational agent** that moves from vague intent → clarifying dialogue → grounded shortlist, using only the SHL Individual Test Solutions catalog.

Key design constraints:
- Stateless API (full history per call)
- Max 8 turns, 30-second timeout per call
- Schema-strict responses (name, url, test_type)
- Zero hallucination: every URL must come from the scraped catalog

---

## Architecture

```
POST /chat → FastAPI → SHLAgent
                           ├── FAISS Retrieval (semantic + keyword)
                           ├── Context Assembly (catalog entries)
                           └── Gemini 1.5 Flash → JSON Response
                                  └── Validation (URL catalog check)
```

**Stack choices:**

| Component | Choice | Reason |
|-----------|--------|--------|
| LLM | Gemini 1.5 Flash | Free tier, 1M context, fast (< 5s typical) |
| Embedding | all-MiniLM-L6-v2 | 384-dim, lightweight, runs on CPU |
| Vector DB | FAISS IndexFlatIP | No external service needed, 377 items fits in-memory |
| Framework | FastAPI | Async, fast, auto-docs, Pydantic validation |
| Deployment | Render.com | Free tier, easy GitHub deploy |

---

## Catalog Ingestion

Scraped **377 Individual Test Solutions** from `shl.com/solutions/products/product-catalog/?type=1` across 32 pages using `requests` + `BeautifulSoup`. Extracted:
- Product name (canonical)
- URL (product page)
- Test type codes (A, B, C, D, E, K, M, P, S)

Catalog stored as `catalog.json` — committed to repo so no cold-start scrape needed.

---

## Retrieval Strategy (RAG)

Two-stage retrieval to maximize recall:

1. **Semantic search** (FAISS): Embed the last 3 user messages → cosine similarity against catalog embeddings. Returns top-15 items above 0.05 threshold.

2. **Keyword fallback**: Token-level matching of query words against product names (score +3 for name match, +1 for description). Appends up to 5 unique keyword hits.

Both stages deduplicate by URL. Merged top-15 fed as context to the LLM.

**Why this works**: Semantic search handles "backend developer" → "Java", "Python" assessments. Keyword search handles exact product name queries like "OPQ32r" or "Verify".

---

## Agent Design

The agent is a **single-prompt LLM call** (not a graph/chain) with explicit decision logic in the system prompt:

```
CLARIFY  → if role unknown and turn ≤ 4
RECOMMEND → if role known + any context OR turn ≥ 5
REFINE   → if user says "add/remove/include"
COMPARE  → if user asks "difference between X and Y"
REFUSE   → if off-topic or injection detected
```

**Turn budget enforcement**: At message count ≥ 5, the system prompt explicitly warns the LLM to provide recommendations now. This prevents exhausting the 8-turn cap on clarifications.

**Response validation pipeline** (defense in depth):
1. Pre-LLM: keyword injection check (fast, no tokens used)
2. Post-LLM: JSON parse → URL validation against catalog
3. Three fallback strategies: exact URL → exact name → fuzzy name match
4. Items that can't be verified against catalog are silently dropped

---

## Prompt Design

The system prompt is injected into the **first user message** (Gemini doesn't have a separate system role in the chat API). Key decisions:

- **Explicit decision tree** in prompt (CLARIFY / RECOMMEND / REFINE / COMPARE / REFUSE)
- **Mandatory JSON format** shown as a code block with rules
- **Catalog context** appended as a dedicated section with exact names and URLs
- **Temperature = 0.3**: Low enough for consistent JSON, high enough for natural language

What didn't work initially:
- Pure semantic search missed exact product name queries → added keyword fallback
- LLM was recommending on turn 1 for vague queries → added explicit rule "do not recommend if query is vague and turn ≤ 1"
- LLM was exhausting turns with questions → added turn budget warning at turn 5

---

## Evaluation Approach

Tested against 6 behavioral probes before submission:

| Probe | Expected | Result |
|-------|----------|--------|
| Vague query ("I need an assessment") | Clarify, recs=[] | ✓ Pass |
| Role + context | Recommend 1-10 | ✓ Pass |
| Refinement ("add personality tests") | Updated shortlist includes P-type | ✓ Pass |
| Off-topic (weather question) | Refuse, recs=[] | ✓ Pass |
| Prompt injection | Refuse, recs=[] | ✓ Pass |
| Turn 6-7 conversation | Recommendations provided | ✓ Pass |

Schema compliance: All responses include `reply` (str), `recommendations` (list), `end_of_conversation` (bool). All recommendation URLs verified against catalog before returning.

**Recall@10 strategy**: Retrieve 15 items (semantic + keyword) → LLM selects the most relevant 1-10. Broader retrieval improves recall; LLM re-ranking improves precision.



## What I Would Improve With More Time

1. **Scrape product descriptions** from individual product pages to enrich embeddings
2. **Fine-tune retrieval**: Use BM25 hybrid search for better keyword-semantic balance
3. **Conversation memory**: Cache embeddings per session to speed up repeated queries
4. **Evaluation harness**: Run the public 10 conversation traces and measure Recall@10 explicitly

---

*Deployment: Render.com (render.yaml included). First `/health` call allows 2 min cold start for sentence-transformer model loading.*
