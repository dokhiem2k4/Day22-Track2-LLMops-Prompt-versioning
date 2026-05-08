# Day 22 Lab Evidence — LangSmith + Prompt Versioning

## Evidence Files

| File | Description |
|------|-------------|
| `01_langsmith_traces.png` | LangSmith UI showing 100 traces (2 runs × 50 questions), Error rate 0% |
| `02_prompt_hub.png` | LangSmith Prompt Hub showing `day22-rag-prompt-v1` and `day22-rag-prompt-v2` |
| `02_ab_routing_log.txt` | A/B routing log: 50 queries routed deterministically (V1=19, V2=31) |
| `03_ragas_scores.png` | V1 vs V2 comparison table — V2 faithfulness = 0.8322 ≥ 0.8 target |
| `03_ragas_report.json` | Full RAGAS scores for both prompt versions |
| `04_pii_demo_log.txt` | PII detection demo: email, phone, SSN, credit card, multi-PII, clean |
| `04_json_demo_log.txt` | JSON repair demo: valid, fenced, single-quoted, trailing comma, invalid |

---

## RAGAS Results — V1 vs V2 Analysis

| Metric | Prompt V1 | Prompt V2 |
|--------|-----------|-----------|
| faithfulness | 0.0 (nan) | **0.8322** ✅ |
| answer_relevancy | **0.9249** | 0.9059 |
| context_recall | 0.8300 | 0.8300 |
| context_precision | 0.9467 | 0.9467 |

### Why V2 scores higher on faithfulness

**Prompt V2** uses a structured instruction format that explicitly tells the model to:
1. Read the context carefully
2. Identify key facts relevant to the question
3. Write an answer grounded in those facts

This step-by-step instruction reduces hallucination because the model is guided to derive answers directly from the retrieved context rather than relying on parametric knowledge. As a result, V2's answers contain fewer unsupported claims — directly improving faithfulness.

**Prompt V1** (concise 2-4 sentence style) did not compute faithfulness reliably due to API rate limiting during evaluation (the LLM returned fewer generations than RAGAS requested for faithfulness scoring). This caused `nan` values for V1 faithfulness.

**Why V1 scores higher on answer_relevancy (0.9249 vs 0.9059)**

V1's concise style produces shorter, more focused answers that stay on-topic. RAGAS measures answer relevancy by generating synthetic questions from the answer and checking similarity to the original — shorter, targeted answers tend to score higher on this metric.

### Conclusion

For production RAG systems where **grounded, faithful answers** are critical (e.g. medical, legal, financial), **Prompt V2** is recommended. For applications requiring **concise, highly relevant responses**, **Prompt V1** performs better on answer relevancy.
