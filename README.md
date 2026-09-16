# GWR Customer Support Semantic RAG Pipeline

A production-ready, zero-leakage Retrieval-Augmented Generation (RAG) customer support agent for **Great Western Railway (`@GWRHelp`)**. The pipeline performs 12-class customer intent classification, dynamic action decision routing (`RESPOND`, `ASK_CLARIFICATION`, `REQUEST_INFORMATION`, `ESCALATE`), FAISS dense vector retrieval, and human-like empathetic response generation with automated LLM-as-a-Judge evaluation.

---

## ⚡ 15-Minute Reproduction Fast Track

All preprocessed datasets and FAISS vector indices are **already included and pre-built** in this repository. You can clone the repo and reproduce the headline results in **under 15 minutes**.

### Headline Results Summary

| Metric / Dimension | Baseline 1 (Trivial) | Baseline 2 (Keyword/Rule) | Our Semantic RAG (`core/myapproch.py`) |
| :--- | :---: | :---: | :---: |
| **Intent Classification Accuracy** | 0.0% (N/A) | 55.6% | **88.0% – 90.5%** |
| **Response Quality (LLM Judge)** | 25.0% | 45.0% | **82.4%** |
| **Retrieval Utility (LLM Judge)** | N/A | N/A | **76.8%** |
| **Multi-Turn Context Retention** | Fails | Fails | **87.5% retention (Turn 2+)** |
| **Zero-Leakage Quarantine** | N/A | N/A | **100% strict holdout (0 leakage)** |
| **Average Latency** | < 1 ms | ~5 ms | **1.2s – 2.5s end-to-end** |

---

## 1. Quickstart & Environment Setup (2 mins)

### Step 1.1: Clone and Create Virtual Environment
Create an isolated Python 3.10+ virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 1.2: Install Dependencies
Install the required packages (minimal, fast, zero unnecessary bloat):

```bash
pip install -r requirements.txt
```

*Packages installed: `sentence-transformers`, `faiss-cpu`, `pandas`, `python-dotenv`.*

### Step 1.3: Configure API Key
Copy the example environment file and add your Google Gemini API key:

```bash
cp .env.example .env
```

Open `.env` and set your key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

---

## 2. Data Cleaning & Golden Set Quarantine

The data processing pipeline transforms raw multi-brand Twitter customer support interactions into clean, chronologically ordered, thread-reconstructed conversations for `@GWRHelp`.

### Fast Path (Skip — Already Done!)
> **Skip Note:** You do **not** need to download raw data or run cleaning to reproduce results. The pre-cleaned datasets are already committed in the `Data/` directory:
> - `Data/gwr_threads.csv`: Cleaned multi-turn `@GWRHelp` conversation threads.
> - `Data/data_retrieval.csv`: Retrieval corpus with golden threads strictly held out.
> - `Data/gwr_golden_review.csv`: Hand-reviewed golden benchmark dataset.

---

### Full Data Reproduction from Scratch (Optional)

If you wish to re-run the entire data cleaning pipeline from raw data:

1. Download `twcs.csv` from Kaggle's [Customer Support on Twitter Dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (~500 MB).
2. Place `twcs.csv` in the `Data/` folder (`Data/twcs.csv`).
3. **Stage 1 — Thread Extraction & Filtering**:
   ```bash
   python data_clean/stage_1.py
   ```
   *What it does:* Filters exclusively for `@GWRHelp` brand tweets, traverses parent-child reply relationships, resolves multi-branch tree structures, normalizes tweet IDs, and exports reconstructed conversations to `Data/gwr_threads.csv`.

4. **Stage 2 — Golden Set Holdout & Quarantine**:
   ```bash
   python data_clean/stage2.py
   ```
   *What it does:* Strictly subtracts all thread IDs present in `Data/gwr_golden_review.csv` from the retrieval candidate pool to guarantee **zero test-set data leakage**, exporting the sanitized corpus to `Data/data_retrieval.csv`.

---

## 3. FAISS Vector Index

Semantic retrieval uses dense embeddings from `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) indexed with FAISS L2 flat indexing.

### Fast Path (Pre-Stored Index)
The index and metadata are already generated and saved in:
- `index/faiss.index` (Dense vector index)
- `index/retrieval_metadata.json` (Thread texts and metadata mapping)

### Rebuilding the Index from Scratch (Optional)
To rebuild the FAISS index:

```bash
python scripts/build_index.py
```

*What it does:*
- Applies sliding-window chunking across long conversation threads.
- Validates golden set quarantine before embedding.
- Embeds corpus chunks in batches and persists the index to `index/faiss.index`.

---

## 4. Running the Support Agent & Speed Testing (`core/myapproch.py`)

`core/myapproch.py` is the primary entry point for querying the RAG pipeline. It retrieves the top-$k$ most relevant historical resolution patterns, identifies customer intent, selects a policy action, and generates an empathetic, context-aware reply.

### 4.1 Quick Interactive / Default Test
Run the built-in interactive test and latency benchmark:

```bash
python core/myapproch.py
```

**Sample Output:**
```text
=================================================================
GWR SUPPORT AGENT - INTERACTIVE TEST & SPEED BENCHMARK
=================================================================
Context: Customer: @GWRHelp my train from Paddington to Reading was delayed by 45 minutes.
Message: How do I claim compensation for this delay?
Top-K:   3

Running pipeline...

=================================================================
RESULT
=================================================================
Intent:     REFUND / COMPENSATION
Decision:   RESPOND
Confidence: 0.95
Latency:    1420.5 ms (1.42 s)

Response:
I am very sorry for the 45-minute delay on your journey from Paddington to Reading today. You are eligible to claim compensation through our Delay Repay scheme. Please visit https://www.gwr.com/delayrepay to submit your claim with your ticket details. If you need any further help, feel free to reach out. - GWRHelp

Retrieved Examples Count: 3
  [1] Thread: GWR_004512 (Score: 0.742)
  [2] Thread: GWR_008104 (Score: 0.718)
  [3] Thread: GWR_001923 (Score: 0.695)
=================================================================
```

### 4.2 Speed Test with Custom Message & Context
Pass your own query or prior conversation context directly via CLI flags:

```bash
python core/myapproch.py --message "Where can I find lost property left at Swansea station?" --context "Customer: @GWRHelp I left my black backpack on the 14:15 train." --top-k 3
```

### 4.3 Python API Integration
You can also import and call the pipeline directly from Python code:

```python
from core.myapproch import run_support_agent

result = run_support_agent(
    conversation_context="Customer: @GWRHelp our train from Bristol was cancelled.",
    customer_message="Can I use my ticket on the next service?",
    top_k=3
)

print("Intent:", result["intent"])
print("Decision:", result["decision"])
print("Response:", result["response"])
```

---

## 5. Evaluation Harness (`core/eval.py`)

The evaluation harness evaluates agent performance against human ground truth using an independent **LLM-as-a-Judge** scoring system.

### 5.1 Evaluation Modes
1. **`replay` (Multi-Turn Sequential Replay — Default)**:
   Sequentially replays the thread turn-by-turn. Turn 1 AI responses are dynamically injected into the conversation context for Turn 2, testing whether the agent maintains conversational state and context retention over time.
2. **`first_turn` (Initial Contact Triage)**:
   Evaluates initial customer root inquiries in isolation.

### 5.2 Reproducing Headline Evaluation Results (< 5 mins)

Run multi-turn replay on 15 sampled threads from the golden set:
```bash
python core/eval.py --input-file Data/gwr_golden_review.csv --samples 15 --eval-mode replay
```

Run single-turn evaluation on 25 samples:
```bash
python core/eval.py --input-file Data/gwr_golden_review.csv --samples 25 --eval-mode first_turn
```

### 5.3 Custom Evaluation CSV Requirements
If you want to evaluate on your own test dataset, your CSV must include these required columns:

| Column Header | Type | Description |
| :--- | :---: | :--- |
| `thread_id` | `str` | Unique thread identifier (e.g. `GWR_000123`). |
| `intent` | `str` | Golden intent label (must match one of the 12 allowed taxonomy categories). |
| `full_conversation` | `str` | Chronological multi-turn transcript with speaker prefixes. |

**Expected `full_conversation` Format:**
```text
Customer: @GWRHelp my train from Paddington to Reading was delayed by 45 minutes.
GWRHelp: Hi there, sorry for the delay. You can claim Delay Repay at gwr.com/delayrepay. - Andy
Customer: Thank you, do I need to send photos of my paper tickets?
GWRHelp: Yes, please upload a clear photo of your paper ticket when submitting the form. - Andy
```

> **Recommendation:** To guarantee proper column names and formatting, sample directly from `Data/gwr_golden_review.csv`, which is already curated and validated.

### 5.4 Evaluation Output Data
All results, judge reasoning, retrieved context, and turn metrics are automatically saved to `Data/evaluation_results.csv`:
- `thread_id`, `turn_number`, `customer_message`, `conversation_context`
- `retrieved_thread_ids`, `retrieval_scores`, `retrieved_context`
- `ground_truth_intent`, `predicted_intent`, `intent_match`
- `predicted_decision`, `predicted_confidence`
- `ground_truth_response`, `generated_response`
- `retrieval_score_percent`, `retrieval_reasoning` (Judge score 0–100%)
- `response_score_percent`, `response_reasoning` (Judge score 0–100%)

---

## 6. Repository Structure

```text
├── Data/
│   ├── gwr_threads.csv                 # Cleaned GWR multi-turn threads
│   ├── data_retrieval.csv              # Sanitized retrieval corpus (golden quarantined)
│   ├── gwr_golden_review.csv           # Golden evaluation benchmark dataset
│   ├── human_evaluation_results.csv    # Human review results
│   └── llm_evaluation_results.csv      # LLM judge benchmark results
├── core/
│   ├── myapproch.py                    # Main RAG agent interface & speed test
│   ├── eval.py                         # Multi-turn sequential replay & judge harness
│   ├── baseline1.py                    # Trivial static response baseline
│   ├── baseline2.py                    # Keyword / rule-based classifier baseline
│   └── prompt.py                       # LLM Judge prompt templates & client
├── data_clean/
│   ├── stage_1.py                      # Filters TWCS dataset to GWR threads
│   └── stage2.py                       # Golden quarantine holdout filter
├── index/
│   ├── faiss.index                     # Persisted FAISS vector index
│   └── retrieval_metadata.json         # Retrieval chunks & thread metadata
├── rag/
│   ├── config.py                       # Paths, taxonomy, decisions, hyperparameters
│   ├── embeddings.py                   # SentenceTransformer embedding wrapper
│   ├── index.py                        # Chunking and FAISS index construction
│   ├── llm.py                          # Gemini client with strict JSON schema validation
│   ├── pipeline.py                     # SupportAgentPipeline orchestration
│   ├── prompt.py                       # Agent system instructions & prompt builder
│   └── retriever.py                    # Semantic retriever & query reformulator
├── scripts/
│   └── build_index.py                  # Script to rebuild FAISS index from scratch
├── .env.example                        # Template environment configuration
├── requirements.txt                    # Project Python dependencies
└── README.md                           # Documentation & quick reproduction guide
```

---

## 7. Allowed Taxonomy Reference

### 12 Supported Intent Categories
1. `BOOKING / TICKETS`
2. `TRAIN DELAY / CANCELLATION`
3. `TRAIN / SERVICE INFORMATION`
4. `REFUND / COMPENSATION`
5. `PAYMENT / CHARGES`
6. `BAGGAGE / LOST PROPERTY`
7. `ACCESSIBILITY / SPECIAL ASSISTANCE`
8. `ROUTE / TIMETABLE / STATION`
9. `STAFF / SERVICE COMPLAINT`
10. `TECHNICAL / WEBSITE / APP`
11. `GENERAL CUSTOMER SERVICE`
12. `OTHER`

### 4 Action Decisions
- `RESPOND`: Sufficient information to provide an immediate direct answer.
- `ASK_CLARIFICATION`: Query is ambiguous; asks targeted clarification questions.
- `REQUEST_INFORMATION`: Details required (booking reference, ticket image, journey date/time).
- `ESCALATE`: Safety concerns, formal complaints, staff misconduct, or special assistance failures.
