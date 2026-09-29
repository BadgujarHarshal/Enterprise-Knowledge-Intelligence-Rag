# Enterprise Knowledge Intelligence RAG

Enterprise Knowledge Intelligence RAG (EKI-RAG) is an evidence-grounded Retrieval-Augmented Generation system designed to answer knowledge-base questions using hybrid retrieval, multi-hop evidence gathering, reranking, citation-aware generation, and systematic evaluation.

The project was developed as a complete end-to-end RAG engineering and evaluation system rather than as a simple LLM chatbot.

## Overview

Traditional RAG systems commonly rely on a single retrieval method followed by direct generation. EKI-RAG explores a more structured pipeline:

```text
Documents
    ↓
Corpus Construction
    ↓
Sentence-Aware Chunking
    ↓
Dense Embeddings
    ↓
FAISS Dense Retrieval
    +
BM25 Sparse Retrieval
    ↓
Reciprocal Rank Fusion
    ↓
Multi-Hop Evidence Retrieval
    ↓
Cross-Encoder Reranking
    ↓
Evidence-Grounded Prompt Construction
    ↓
LLM Generation
    ↓
Citation Extraction
    ↓
Evaluation
    ↓
FastAPI
    ↓
React Frontend

Project Goals

The main goals of EKI-RAG were:

Build a production-oriented RAG architecture.
Evaluate dense retrieval against sparse retrieval.
Build a hybrid retrieval system.
Introduce multi-hop evidence retrieval.
Evaluate cross-encoder reranking.
Generate answers using supplied evidence.
Track answer and citation quality separately.
Perform systematic RAG failure attribution.
Optimize the system using measured evaluation results.
Expose the system through FastAPI.
Provide a React-based user interface.
Preserve reproducibility through frozen datasets and evaluation populations.
Dataset

The primary evaluation dataset is the HotpotQA distractor configuration.

The project uses:

HotpotQA training corpus for the retrieval corpus.
Official HotpotQA validation data for held-out evaluation.
A locked fully-evaluable subset of 560 questions for the main retrieval and generation experiments.
Dataset Statistics
Property	Value
Training examples	90,447
Validation examples	7,405
Development subset	1,000
Unique training documents	482,021
Canonical chunks	537,559
Fully evaluable evaluation questions	560
Partially evaluable questions	332
Non-evaluable questions	108

The canonical chunk schema is:

chunk_id
document_id
title
content
source
chunk_index
metadata

The project deliberately preserves content as the canonical chunk text field.

Architecture
1. Corpus Construction

The HotpotQA training corpus is converted into a canonical document representation.

Each document contains its title, source information, and content.

The resulting corpus contains 482,021 unique training documents.

2. Sentence-Aware Chunking

Documents are split into sentence-aware chunks.

The chunking configuration uses:

Target chunk size: approximately 800 characters
Maximum chunk size: 1200 characters
Sentence overlap: 1 sentence

The final canonical corpus contains:

537,559 chunks
3. Dense Retrieval

Dense retrieval uses:

sentence-transformers/all-MiniLM-L6-v2

Embedding dimension:

384

Embeddings are normalized and indexed using FAISS.

The final FAISS index contains:

537,559 vectors
4. BM25 Retrieval

BM25 provides lexical retrieval alongside dense semantic retrieval.

The implementation uses:

bm25s

with the configured BM25 parameters:

k1 = 1.5
b  = 0.75
5. Hybrid Retrieval

Dense and BM25 retrieval are combined using weighted Reciprocal Rank Fusion.

The selected configuration is:

Dense weight = 1.0
BM25 weight  = 4.0
RRF k        = 60
Candidate K  = 100

The hybrid configuration was selected through a retrieval ablation study rather than being chosen arbitrarily.

6. Cross-Encoder Reranking

The project evaluates:

cross-encoder/ms-marco-MiniLM-L6-v2

The reranker operates on the hybrid candidate set.

The default configuration uses:

Candidate K = 100
Output K    = 20
Maximum length = 512
Batch size = 32
7. Multi-Hop Retrieval

The multi-hop retriever performs evidence-guided retrieval in multiple hops.

The implemented strategy is:

Question
   ↓
Hybrid Retrieval
   ↓
Hop 1 Evidence
   ↓
Evidence-Guided Query Expansion
   ↓
Hop 2 Retrieval
   ↓
Deduplication
   ↓
Score Aggregation
   ↓
Final Evidence

Configuration:

Maximum hops = 2
Candidates per hop = 20
Expansion seeds = 3
Hop decay = 0.8

The system is intentionally deterministic and does not require an LLM to perform the retrieval expansion step.

8. Evidence-Grounded Generation

The generation layer contains:

Prompt construction
LLM abstraction
Answer generation
Citation extraction
Citation-context mapping

The prompt instructs the model to:

Use only supplied evidence.
Avoid inventing unsupported facts.
State when the evidence is insufficient.
Associate factual claims with citation identifiers.

The evaluated generation model was:

Qwen/Qwen2.5-3B-Instruct
9. Citation Handling

The system extracts citations from generated answers and maps them back to the supplied evidence context.

Citation information includes:

citation_id
title
content
chunk_id
hop
score

Citation evaluation is kept separate from answer correctness so that an answer can be analyzed independently from whether its supporting evidence was correctly cited.

Evaluation

The project uses multiple evaluation layers.

Retrieval Metrics
Hit@K
Supporting Coverage@K
MRR

The primary retrieval evaluation uses the locked 560-question fully-evaluable population.

Generation Metrics
Exact Match
Answer F1
Answer correctness
Evidence overlap
Context title recall
Citation Metrics
Citation coverage
Citation title precision
Citation title recall
Citation validity
System Metrics

The architecture also tracks:

Retrieval latency
Generation latency
Token usage
Runtime
Model configuration
Candidate configuration
Retrieval Results

The locked dense retrieval baseline was evaluated on the fully-evaluable 560-question population.

Dense Baseline
Hit@5       = 0.1250
Coverage@5  = 0.0670
MRR@5       = 0.1146

Hit@10      = 0.1339
Coverage@10 = 0.0732
MRR@10      = 0.1158

Hit@20      = 0.1393
Coverage@20 = 0.0768
MRR@20      = 0.1162
Hybrid Retrieval

The selected hybrid configuration produced:

Dense weight = 1.0
BM25 weight  = 4.0
RRF k        = 60
Candidate K  = 100

Results on the same locked population:

Hit@5       = 0.8589
Coverage@5  = 0.5250
MRR@5       = 0.6361

Hit@10      = 0.9214
Coverage@10 = 0.6098
MRR@10      = 0.6444

Hit@20      = 0.9607
Coverage@20 = 0.6786
MRR@20      = 0.6472
Multi-Hop Retrieval

The Phase 6 multi-hop evaluation produced:

Hit@5       = 0.8286
Coverage@5  = 0.5545
MRR@5       = 0.5776

Hit@10      = 0.9018
Coverage@10 = 0.6420
MRR@10      = 0.5875

Hit@20      = 0.9589
Coverage@20 = 0.7527
MRR@20      = 0.5917

These retrieval metrics should not be directly interpreted as generation-quality metrics. Retrieval success and answer quality are evaluated separately.

Phase 9.5 Error Analysis

The final error-analysis population contains:

Questions: 560
Correct answers: 241
Incorrect answers: 319

Retrieval complete: 214
Retrieval incomplete: 346

Abstentions: 70

Successful RAG: 129
Retrieval failures: 169
Generation failures: 80
Lucky answers: 112

Appropriate abstentions: 65
Over-conservative abstentions: 5

Unsupported-generation candidates: 25
Citation failures: 490

Mean metrics:

Answer F1              = 0.3964
Exact Match             = 0.3143
Context Title Recall    = 0.6420
Evidence Overlap        = 0.6752
Citation Coverage       = 0.0000
Citation Title Precision= 0.5035
Citation Title Recall   = 0.4277
Citation Validity       = 0.0000

These results exposed an important distinction in the system:

Retrieval quality
≠
Generation quality
≠
Citation integrity

The project therefore does not treat a correct final answer as sufficient evidence that the RAG pipeline behaved correctly.

Phase 10 Optimization

Phase 10 performs evidence-based configuration analysis and regression checking.

The optimizer selected the existing measured configuration:

baseline_qwen2_5_3b

as the measured candidate.

The final report explicitly records areas that were not experimentally optimized:

retrieval_optimization
generation_optimization
citation_integrity_optimization

The optimizer therefore does not claim improvements that were not experimentally measured.

Backend

The project exposes the RAG system through FastAPI.

Main backend structure:

app/
├── api/
│   ├── __init__.py
│   ├── schemas.py
│   └── routes.py
│
├── services/
│   ├── __init__.py
│   └── rag_service.py
│
└── main.py

API endpoints:

GET  /
GET  /api/health
POST /api/query

Example request:

{
  "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
  "top_k": 10
}
Frontend

The project includes a React/Vite frontend.

Architecture:

React
   ↓
API Service
   ↓
FastAPI
   ↓
RAG Service

Frontend structure:

frontend/
├── package.json
├── vite.config.js
├── index.html
├── .env
├── .env.example
└── src/
    ├── main.jsx
    ├── App.jsx
    ├── styles/
    │   └── global.css
    ├── services/
    │   └── api.js
    ├── hooks/
    │   └── useRagQuery.js
    ├── utils/
    │   └── ragResponseNormalizer.js
    └── components/
        ├── Header.jsx
        ├── QueryInput.jsx
        ├── AnswerPanel.jsx
        ├── CitationList.jsx
        ├── CitationCard.jsx
        ├── SourceEvidence.jsx
        ├── QueryHistory.jsx
        ├── EmptyState.jsx
        ├── LoadingState.jsx
        └── ErrorState.jsx

The frontend production build completes successfully with Vite.

Local Development
Backend

Create and activate the Conda environment:

conda activate eki-rag

Start FastAPI:

uvicorn app.main:app --reload

Backend:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

Health check:

http://127.0.0.1:8000/api/health
Frontend

Open a second terminal:

cd frontend
npm install
npm run dev

Frontend:

http://localhost:5173
Production Frontend Build
npm run build
Runtime Artifacts

The local serving layer expects the frozen retrieval artifacts:

data/
└── processed/
    └── hotpotqa/
        └── hotpotqa_chunks.jsonl

models/
├── embeddings/
│   └── hotpotqa_dense.index
│
└── retrieval/
    └── hotpotqa_bm25/

The canonical Google Drive artifact locations used during experimentation were:

/content/drive/MyDrive/EKI-RAG/datasets/hotpotqa_chunks.jsonl

/content/drive/MyDrive/EKI-RAG/faiss/hotpotqa_dense.index

/content/drive/MyDrive/EKI-RAG/bm25/hotpotqa_bm25/

The retrieval artifacts are large and are not intended to be committed to GitHub.

Local LLM Requirement

The evaluated generation model is:

Qwen/Qwen2.5-3B-Instruct

The model is substantially larger than the lightweight retrieval models and requires additional local storage for inference.

The complete evaluated Qwen 3B runtime is therefore not bundled in this repository.

The retrieval and evaluation artifacts remain the authoritative experimental results.

For environments with limited disk space, the generation model should be hosted separately or replaced with a smaller model only as a separate demonstration configuration. A smaller model should not be treated as equivalent to the evaluated Phase 7/10 configuration.

Reproducibility

The project follows a frozen-artifact approach.

The following were reused rather than regenerated during later phases:

HotpotQA corpus
Chunk corpus
Dense embeddings
FAISS index
BM25 index
Frozen 560-question evaluation set

This prevents later experiments from silently changing the underlying retrieval population.

Testing

The project contains tests covering the major components.

Run the complete Python test suite:

pytest -q

The finalized project test suite reached:

179 passed

The frontend production build can be verified with:

cd frontend
npm run build
Project Structure
Enterprise Knowledge Intelligence Rag/
│
├── app/
│   ├── api/
│   ├── core/
│   ├── evaluation/
│   ├── generation/
│   ├── ingestion/
│   ├── models/
│   ├── reranking/
│   ├── retrieval/
│   └── services/
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── evaluation/
│   └── sample/
│
├── models/
│   ├── embeddings/
│   ├── reranker/
│   └── llm/
│
├── notebooks/
│
├── scripts/
│
├── tests/
│   ├── ingestion/
│   ├── retrieval/
│   ├── reranking/
│   ├── generation/
│   └── evaluation/
│
├── frontend/
│
├── docker/
│
├── configs/
│
└── README.md
Key Engineering Decisions
Hybrid Retrieval Instead of Dense Retrieval Alone

Dense retrieval and BM25 capture different retrieval behavior.

The project therefore evaluates both and combines them using RRF rather than assuming one retrieval method is sufficient.

Multi-Hop Retrieval Without an LLM

The multi-hop retrieval stage is evidence-guided and deterministic.

This separates retrieval reasoning from generation and makes the retrieval stage easier to evaluate independently.

Frozen Evaluation Population

The 560-question fully-evaluable population is kept fixed across the main retrieval experiments.

This prevents changing evaluation populations from making different experiments appear directly comparable when they are not.

Separate Citation Evaluation

Citation correctness is measured separately from answer correctness.

A model can produce a correct answer while failing to provide valid supporting citations.

Error Attribution

The project distinguishes:

Retrieval Failure
Generation Failure
Lucky Answer
Appropriate Abstention
Over-Conservative Abstention
Unsupported Generation
Citation Failure

This makes the system easier to debug than evaluating only final answer accuracy.

Limitations

The project has several important limitations.

Dataset

HotpotQA is a benchmark dataset and does not represent every enterprise knowledge-base environment.

Retrieval Corpus

The retrieval corpus is derived from the HotpotQA training corpus.

Synthetic/Benchmark Evaluation

The evaluation population is benchmark-based rather than an internal enterprise knowledge base.

Citation Evaluation

Citation evaluation is based on the available structured evidence and supporting-title information. Title-level recall should not be interpreted as complete sentence-level factual attribution.

Generation

The evaluated Qwen 3B model is a pretrained instruction model. The project does not train an LLM from scratch.

Local Deployment

The full evaluated generation model requires substantially more local storage than the retrieval components.

Production Security

The current project is a research/engineering prototype. Production deployment would additionally require:

Authentication
Authorization
Rate limiting
Secrets management
Observability
Persistent query logging
Document-level access control
Data encryption
Model serving infrastructure
Production database integration
What This Project Demonstrates

This project demonstrates practical experience with:

Python
Machine Learning
Information Retrieval
Dense Retrieval
BM25
FAISS
RRF
Multi-Hop Retrieval
Cross-Encoder Reranking
RAG
LLM Applications
Prompt Engineering
Citation Grounding
Evaluation
Error Attribution
FastAPI
React
Vite
Docker/Kubernetes Architecture

More importantly, the project demonstrates the ability to evaluate a RAG system as a collection of measurable components rather than treating the LLM response as the only metric.

Status
Phase 1 — Dataset & Corpus Construction       COMPLETE
Phase 2 — Chunking                            COMPLETE
Phase 3 — Dense Retrieval                    COMPLETE
Phase 4 — Hybrid Retrieval                   COMPLETE
Phase 5 — Cross-Encoder Reranking             COMPLETE
Phase 6 — Multi-Hop Retrieval                COMPLETE
Phase 7 — LLM Generation                     COMPLETE
Phase 8 — RAG Evaluation                     COMPLETE
Phase 9 — Error Analysis                     COMPLETE
Phase 10 — Optimization & Regression         COMPLETE
Frontend                                      COMPLETE
FastAPI Integration                           COMPLETE
Local Full 3B Inference                       HARDWARE-LIMITED
Final Project State

The experimental and evaluation pipeline is frozen at Phase 10.

The final measured configuration is:

Hybrid Retrieval
    +
Multi-Hop Evidence Retrieval
    +
Qwen 2.5 3B Generation
    +
Citation-Aware Evaluation

The local React and FastAPI layers provide the application interface around the frozen RAG architecture.

The full Qwen 3B local inference environment requires additional disk capacity and is therefore treated as a deployment requirement rather than part of the lightweight source repository.

License

This repository contains project code and configuration created for the EKI-RAG project.

Dataset, pretrained models, and third-party libraries remain subject to their respective licenses and terms.


Enterprise-Knowledge-Intelligence-RAG/
│
├── app/
│   ├── api/
│   ├── core/
│   ├── evaluation/
│   ├── generation/
│   ├── ingestion/
│   ├── models/
│   ├── reranking/
│   ├── retrieval/
│   └── services/
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│       └── small final reports
│
├── scripts/
│
├── tests/
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   ├── index.html
│   └── .env.example
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   ├── nginx.conf
│   └── docker-compose.yml
│
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md