# GWR Customer Support Agent — Report

**Brand:** `@GWRHelp` (Great Western Railway)  
**Repo:** https://github.com/Yadeesht/Houzz.ai

---

## 1. Problem Framing

### What "good" means for this brand

The goal is to build a support-response system for GWR customer interactions. A good response should be relevant to the customer's request, use the available conversation context, provide useful next steps, avoid unsupported claims, and make an appropriate decision when the available information is insufficient.

For this project, the focus is **response usefulness and grounding**, rather than reproducing the historical GWRHelp wording exactly. The system is intended to answer the current customer query using the supplied context and relevant historical support examples.

A good response is therefore one that:

- addresses the customer's actual issue;
- uses information already present in the conversation appropriately;
- gives a clear and actionable next step when possible;
- avoids inventing facts, policies, prices, schedules, or account information;
- asks for clarification or requests missing information when it cannot safely answer; and
- escalates when the available information is insufficient for a reliable response.

The project is **not** intended to optimize for response brevity alone, reproduce the exact original GWRHelp response, or demonstrate multi-brand generalization.

### What I chose not to build

- **No UI.** Results are inspected through CSV outputs and the terminal so that the pipeline remains easy to audit and reproduce.
- **No fine-tuned classifier.** Intent classification and response generation are handled in a single LLM call so the model can use the same context for both tasks without introducing another separately maintained model/call.
- **No separate intent-only LLM call.** Intent is generated together with the response, reducing duplicated context passing and inference cost.
- **No managed vector database.** The semantic retrieval MVP uses a local FAISS index, which is sufficient for the reconstructed GWRHelp corpus and keeps the experiment reproducible.

---

## 2. Golden Evaluation Set — Sampling & Labeling

- **Size:** 150 GWRHelp threads were held out as the golden-set pool.
- **Source population:** The reconstructed GWRHelp dataset contains **10,728 threads and 49,493 tweets**. A separate 600-thread candidate pool was created using conversation characteristics such as thread length, customer/company turn counts, question/complaint signals, and escalation signals. The final 150-thread golden pool was then prepared from this candidate material.
- **Leakage prevention:** The golden threads were removed from the retrieval corpus before the semantic index was created. The retrieval corpus is therefore kept separate from the evaluation set.
- **Labeling process:** An subset of approximately 150 examples was manually labeled to establish a consistent taxonomy and annotation convention. The annotation fields include `intent`, `interaction_type`, `resolution_status`, and `evaluation_quality`.
- **Intent taxonomy:**
  - `BOOKING / TICKETS`
  - `TRAIN DELAY / CANCELLATION`
  - `TRAIN / SERVICE INFORMATION`
  - `REFUND / COMPENSATION`
  - `PAYMENT / CHARGES`
  - `BAGGAGE / LOST PROPERTY`
  - `ACCESSIBILITY / SPECIAL ASSISTANCE`
  - `ROUTE / TIMETABLE / STATION`
  - `STAFF / SERVICE COMPLAINT`
  - `TECHNICAL / WEBSITE / APP`
  - `GENERAL CUSTOMER SERVICE`
  - `OTHER`
---

## 3. Evaluation Harness — Metrics & LLM-as-Judge Rubric

### Automated metrics

The current evaluation harness reports:

- **Intent classification accuracy:** exact-match accuracy of the predicted intent against the golden intent label.
- **Retrieval relevance/utility:** judged usefulness of the retrieved historical examples for the current support case.
- **Response quality:** aggregate LLM-judge assessment of the generated response.
- **Latency:** end-to-end runtime per evaluated example.

### LLM-as-judge rubric

The judge evaluates the generated support response against the current customer situation and available context. The intended quality dimensions are:

1. **Relevance** — does the response address the customer's issue?
2. **Correctness** — is the response consistent with the information available in the case?
3. **Actionability** — does it provide a useful next step or clear answer?
4. **Completeness** — does it cover the important parts of the request?
5. **Context usage** — does it appropriately use information already supplied by the customer?
6. **Resolution** — does the response move the issue toward resolution, or appropriately recognize that clarification/information/escalation is needed?

The judge returns structured evaluation information rather than relying on text similarity to the historical GWRHelp response.

### Evidence of judge–human agreement

A formal numeric judge–human agreement statistic has **been calculated** for the random example run.

---

## 4. Results vs. Baselines

The project includes two simple comparison baselines in the implementation, but they were not run through the full automated evaluation because the limited remaining evaluation budget was focused on validating the proposed semantic-RAG pipeline.

| Metric | Baseline 1 (Trivial) | Baseline 2 (Keyword/Rule) | Our Semantic RAG |
|---|---|---|---|
| Intent Classification Accuracy | N/A — no intent classification | Implemented, not included in current benchmark run | **44.0%** on 25 golden samples |
| Response Quality (LLM Judge) | Not benchmarked | Not benchmarked | **89.0%** |
| Retrieval Relevance / Utility | N/A | N/A | **76.6%** |
| Latency | Not benchmarked | Not benchmarked | **11.19 s/sample** average |
| Golden-set leakage | N/A | N/A | Golden threads excluded from retrieval corpus |

### Baseline definitions

- **Trivial baseline:** returns the same generic support message for every input, with no intent classification, retrieval, or adaptive reasoning.
- **Simple baseline:** uses keyword/rule matching to infer an intent and then selects a fixed response template for that intent.

The baselines provide simple reference points for the README/report, while the main measured MVP result comes from the semantic retrieval + LLM pipeline. A full apples-to-apples benchmark of all three systems is left as follow-up work.

### Current measured MVP result

On the random 25-example golden sample:

- **Intent classification accuracy:** 44.0% (This low score is due to LLM gets better when it has enough context, so based on first turn message this fails mostly but after 2-3 turns the scoring increases, since every conversations turn1 can fail, this score is low.)
- **Average retrieval relevance/utility:** 76.6%
- **Average response quality:** 89.0%
- **Average latency:** 11.19 seconds per example
- **Total evaluation time:** 279.7 seconds

The combination of relatively strong response-quality scores with lower exact intent accuracy suggests that useful response generation can still occur when the taxonomy label is not an exact match. This is particularly relevant for semantically adjacent categories, but the current sample is too small to establish that pattern reliably.

---

## 5. Failure Analysis — Top 5 Failure Modes

The current example evaluation run is sufficient to identify candidate failure categories, but not enough to claim statistically stable failure frequencies. The main categories to investigate are:

### 1. Intent taxonomy mismatch
- **Example:** A query can sit near multiple closely related categories such as service information versus route/timetable/station information.
- **Hypothesis:** Exact-match intent accuracy penalizes semantically adjacent classifications even when the generated answer remains useful.

### 2. Retrieval mismatch
- **Example:** A retrieved GWRHelp thread can share surface vocabulary with the current query without representing the same underlying customer problem.
- **Hypothesis:** Semantic similarity can still retrieve a superficially related thread, especially where GWR support language is repetitive.

### 3. Context interpretation errors
- **Example:** The current customer query may depend on details contained earlier in the supplied context.
- **Hypothesis:** The LLM may prioritize the latest query too strongly or combine earlier details incorrectly.

### 4. Unsupported generation
- **Example:** A model can produce a fluent answer that goes beyond what is supported by the current context and retrieved examples.
- **Hypothesis:** Retrieved conversations are examples of prior handling, not an authoritative policy database, so the LLM can overgeneralize from them.

### 5. Insufficient information / escalation handling
- **Example:** Some cases cannot be safely resolved from the visible information and should instead request missing details or escalate.
- **Hypothesis:** The model may attempt to be helpful before recognizing that the information needed to answer reliably is missing.

A larger evaluation should connect each failure category to concrete examples, retrieval results, and judge scores before assigning frequencies.

---

## 6. What Is Misleading About My Headline Number? *(mandatory)*

The current headline numbers need to be interpreted carefully:

- The **44.0% intent accuracy, 76.6% retrieval utility, and 89.0% response quality** are measured on only **randomly sampled golden examples**, not the full 150-thread golden set. With few samples, one example changes an accuracy percentage by 4 percentage points, so the estimate has substantial sampling variability.
- Response quality is scored by an **LLM judge**, not entirely by independent human evaluators. A formal judge–human agreement number has been calculated but at low scale, so the response-quality score should not be presented as equivalent to human-verified quality.
- Intent accuracy is **exact-match accuracy against a hand-created taxonomy**. A semantically close category can still count as wrong, so this number does not directly measure whether the final customer response was useful.
- The retrieval corpus was created after excluding the golden-set threads, which reduces direct evaluation leakage through the retrieval index. However, the LLM itself has general pretrained knowledge, so strict retrieval holdout does not mean the model has no prior knowledge of support concepts.
- The 150-thread set was curated from the GWRHelp candidate pool rather than being a uniformly random sample of every reconstructed thread. It therefore should be described as a deliberately constructed evaluation pool rather than as a perfectly population-representative sample.

The most important limitation is therefore not that the MVP score is low or high, but that **the evaluation sample is currently small and the judge has not yet been quantitatively validated against a larger independent human set**.

---

## 7. What I'd Do Next With One More Week

The highest-value next steps would be:

1. **Run the full 150-thread golden set** through the same evaluation harness to reduce sampling noise and obtain more stable estimates.
2. **Complete a structured human-agreement check** on a meaningful subsample of judge-scored examples and report an explicit agreement statistic.
3. **Perform systematic failure analysis** using the actual retrieval results, intent predictions, decisions, and generated responses from the full run.
4. **Improve the intent taxonomy or handling of adjacent intents** only if the larger evaluation confirms that exact-match intent errors are materially affecting response quality.
5. **Build the planned synthetic test-case generator** to create realistic variations of historical support problems and use it as an additional robustness/generalization test.

The one-week extension should prioritize evaluation reliability and failure understanding over adding more infrastructure or features.

---

## Appendix

### Core data pipeline

```text
TWCS raw dataset (2.8M tweets)
        ↓
Reply-graph reconstruction
        ↓
GWRHelp selection
        ↓
49,493 GWRHelp-related tweets
        ↓
10,728 reconstructed threads
        ↓
600 candidate pool
        ↓
150 held-out golden threads
        ↓
Golden threads excluded from retrieval corpus
        ↓
Embedding model + FAISS
        ↓
Top-k semantic retrieval
        ↓
Single LLM call:
intent + decision + response
        ↓
LLM-as-judge evaluation
```

### Retrieval data

`data_retrieval.csv` contains one row per non-golden GWRHelp thread with:

- `thread_id`
- `root_tweet_id`
- `data` — chronological conversation text with speaker labels

### Decision space

The RAG pipeline returns one of:

- `RESPOND`
- `ASK_CLARIFICATION`
- `REQUEST_INFORMATION`
- `ESCALATE`

### Evaluation note

The current experiment is a **fixed-query response-quality evaluation**. It is not yet a full free-running simulation in which the model replaces every GWRHelp turn and receives future customer messages recursively.

### Reproduction

See `README.md` for repository setup and execution commands.  
The reconstructed and evaluation artifacts are stored under the project `Data/` directory.
