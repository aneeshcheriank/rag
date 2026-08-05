# RAG: Parent Document Retrieval for Financial Q&A

A Retrieval-Augmented Generation (RAG) pipeline that answers questions over Apple's 10-K filings, comparing **standard vector search** against **Parent Document Retrieval (PDR)**. Built with LangChain, evaluated with Ragas.

## Motivation

Standard RAG pipelines split documents into small chunks for embedding and retrieval. This works well for simple fact lookups but breaks on complex queries where the answer depends on broader document context (multi-part questions, narrative disclosures, cross-references between tables and footnotes).

Parent Document Retrieval addresses this by using a **two-level chunking strategy**: small "child" chunks (250 tokens) for precise dense retrieval, and larger "parent" chunks (2,000 tokens) that provide the full surrounding context to the LLM at generation time.

## Architecture

```
                    ┌──────────────────────────────┐
                    │         Apple 10-K PDF        │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │    RecursiveCharacterText    │
                    │    Splitter (two levels)      │
                    └──────┬──────────────┬────────┘
                           │              │
                   Child Chunks     Parent Chunks
                   (250 tokens)     (2,000 tokens)
                           │              │
                    ┌──────▼──────┐  ┌───▼──────────┐
                    │   ChromaDB  │  │    Redis      │
                    │  (vectors)  │  │  (full text)  │
                    └──────┬──────┘  └───┬──────────┘
                           │              │
                    ┌──────▼──────────────▼──────────┐
                    │   ParentDocumentRetriever       │
                    │   1. Query → find top-k child   │
                    │   2. Map children → parents     │
                    │   3. Return parent documents    │
                    └──────────────┬─────────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │     DeepSeek V4 Flash         │
                    │     + RAG Prompt              │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │         Answer                │
                    └──────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Orchestration** | LangChain 1.3 |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (384-dims) |
| **Vector Store** | ChromaDB (persistent disk-backed) |
| **Document Store** | Redis (parent full-text storage) |
| **LLM** | DeepSeek V4 Flash (`deepseek-v4-flash`) |
| **Evaluation** | Ragas 0.4.x (context precision, context recall, faithfulness, answer relevancy) |

## Project Structure

```
rag/
├── main.py                  # Interactive Q&A loop
├── load_data.py             # Ingest PDF → build vector store + docstore
├── evaluate.py              # Run evaluation with Ragas metrics
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Redis for document store
├── data/
│   ├── 10K.pdf              # Apple 2025 10-K filing
│   └── evaluation_dataset.json  # 25 curated Q&A pairs
├── src/
│   ├── config.py            # Constants (model names, paths, top-k)
│   ├── model.py             # Embedding & LLM factories
│   ├── data_process.py      # PDF loader & text splitters
│   ├── vector_db.py         # Build parent-document store (ingestion)
│   ├── retriever.py         # Load persisted retriever (inference)
│   ├── pipeline.py          # RAG chain: retrieve → prompt → generate
│   └── prompt.py            # Chat prompt template
├── chroma_db/               # Persisted ChromaDB embeddings
├── models/                  # Cached embedding model
└── explorations/            # Ad-hoc test scripts
```

## Setup

### 1. Install Dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Redis (for parent document storage)

```bash
docker compose up -d
```

### 3. Set your DeepSeek API Key

```bash
export DEEPSEEK_API_KEY="sk-..."
# or add to a .env file in the project root
```

### 4. Ingest Documents

Place an Apple 10-K PDF at `data/10K.pdf`, then:

```bash
python load_data.py
```

This splits the PDF into child/parent chunks, embeds the children into ChromaDB, and stores parent full texts in Redis.

### 5. Ask Questions

```bash
python main.py
```

Type `exit` to quit.

### 6. Run Evaluation

```bash
python evaluate.py
```

Runs the full evaluation dataset through the pipeline and computes Ragas metrics (context precision, context recall, faithfulness, answer relevancy).

## Evaluation

The evaluation dataset contains **25 curated question–answer pairs** spanning 22 categories, from simple fact lookups to multi-hop reasoning and regulatory detail extraction. Each question targets a specific retrieval challenge.

### Results

#### Baseline: Standard Vector DB Retriever

A traditional dense-retrieval pipeline where the same small chunks used for embedding are returned directly as context.

| Metric | Mean |
|--------|------|
| Context Precision | 0.413 |
| Context Recall | 0.385 |
| Faithfulness | 0.673 |
| Answer Relevancy | 0.640 |

#### Parent Document Retriever

Child chunks are embedded and used for similarity search, but their **parent documents** (larger context windows) are returned to the LLM.

| Metric | Mean | Median |
|--------|------|--------|
| Context Precision | **0.783** | 1.000 |
| Context Recall | **0.816** | 1.000 |
| Faithfulness | **0.932** | 1.000 |
| Answer Relevancy | **0.807** | 0.899 |

### What the Results Tell Us

The **median scores of 1.00** across precision, recall, and faithfulness mean that for the majority of questions, the Parent Document Retriever returned exactly the right context and produced a faithful answer. The gap between mean and median reveals that failures are concentrated in a handful of the hardest queries.

#### Where the Baseline Vector Retriever Failed

The baseline succeeded on simple fact lookups (e.g., "What was Apple's diluted EPS?") but collapsed on queries that required broader context:

- **Outright Retrieval Collapse (Zero Recall & Precision)** — On headcount queries ("How many people worked at Apple?") and fiscal calendar questions, dense embeddings prioritized semantic similarity over specific disclosures. Standard chunking severed these notes from the main tables, yielding zero relevant context.

- **Context Fragmentation** — Accounting footnotes, risk factors, and tax commentary sit apart from their related numerical tables. The baseline pulled the numbers but missed the explanatory text entirely, resulting in precision and recall as low as 0.00–0.50.

- **Multi-Hop Gaps** — Questions requiring information from multiple document sections (e.g., computing Services revenue as a percentage of total, then comparing across years) failed when retrieval surfaced only one of the two required tables.

#### Why Parent Document Retrieval Wins

By decoupling the chunk used for **search** (small, precise) from the chunk used for **generation** (large, contextual), PDR ensures the LLM sees the full picture — tables alongside their footnotes, metrics alongside their narrative explanation. The median scores of 1.00 confirm this works for most questions; the remaining failures point to cases where even 2,000-token parent chunks are insufficient (e.g., cross-page relationships).

## Configuration

Key settings in [src/config.py](src/config.py):

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Dense embedding model |
| `LLM_MODEL` | `deepseek-v4-flash` | Generation model |
| `TOP_K` | 2 | Child chunks to retrieve per query |
| `CHROMA_STORAGE` | `./chroma_db` | Persistence directory for vector DB |

The child/parent split sizes are configured in `get_splitter()` in [src/data_process.py](src/data_process.py):
- **Child splitter**: 250 tokens with 50-token overlap
- **Parent splitter**: 2,000 tokens with 200-token overlap

## Known Limitations & Next Steps

1. **Cross-page relationships** — Some answers span sections too far apart for even 2,000-token parent chunks. Graph-based retrieval or summarization chains could bridge these gaps.
2. **TOP_K sensitivity** — The current setting of `k=2` is tuned for precision; higher recall tasks may benefit from larger k at the cost of context noise.
3. **Chunk boundary artifacts** — Recursive character splitting can still cut through the middle of tables. Table-aware splitters may improve results further.
4. **LlamaIndex comparison** — The exploration scope includes comparing LangChain vs. LlamaIndex retrieval pipelines (in progress).
