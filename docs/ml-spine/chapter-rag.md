---
layout: default
title: "ML-A · RAG in Depth"
parent: "ML Spine"
nav_order: 1
description: "Retrieval-Augmented Generation end to end: chunking, embedding, indexing, retrieval, reranking, evaluation, and production architecture."
---

# RAG in Depth
{: .no_toc }

*ML Spine · Chapter A*
{: .text-grey-dk-000 .fs-4 }

<div class="reading-meta">
  ⏱ 35 min read &nbsp;·&nbsp;
  🟩 ML: L3–L4 &nbsp;·&nbsp;
  🟦 DSA inside: Heap · HashMap · DP · Graph
</div>

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. What RAG actually is

RAG stands for Retrieval-Augmented Generation. The one-line definition: **give the LLM the right context before asking it to generate.**

Without RAG, an LLM can only answer from what it learned during training. That means it hallucinates recent facts, doesn't know your private data, and can't cite sources. RAG fixes all three.

The architecture is simple in concept:

```
User query
    ↓
[Retriever] → finds K relevant chunks from a corpus
    ↓
[Prompt builder] → stuffs query + chunks into context window
    ↓
[LLM] → generates answer grounded in retrieved chunks
    ↓
Answer (with citations)
```

Every component of this pipeline has algorithmic depth. This chapter covers each one.

---

## 2. The full RAG pipeline

```
Raw documents
    ↓
[1. Chunking]         → split documents into retrievable pieces
    ↓
[2. Embedding]        → convert chunks to dense vectors
    ↓
[3. Indexing]         → store vectors for fast retrieval
    ↓
                    ← (offline, done once)
                    ← (online, per query)
Query
    ↓
[4. Query embedding]  → same embedding model as step 2
    ↓
[5. Retrieval]        → find K nearest chunks
    ↓
[6. Reranking]        → reorder retrieved chunks by relevance
    ↓
[7. Prompt building]  → pack query + chunks into context
    ↓
[8. Generation]       → LLM produces answer
    ↓
[9. Evaluation]       → measure retrieval + generation quality
```

Each step has failure modes. Each failure mode has a metric. Knowing both is what separates a good RAG implementation from a production-ready one.

---

## 3. Chunking

### Why chunking matters

The embedding model has a token limit (typically 512–8192 tokens). Chunks must fit. But chunk size also controls retrieval quality:

- **Too large:** chunk contains too much irrelevant content → low precision, noise in context
- **Too small:** chunk loses surrounding context → low recall, incomplete answers

### Chunking strategies

**Fixed-size chunking** — split every N tokens with M token overlap:

```python
def chunk_fixed(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Simplest chunking. Works well for uniform text (articles, reports).
    Overlap preserves context across chunk boundaries.
    
    DSA inside: sliding window over tokens.
    """
    tokens = text.split()   # in practice: use a tokenizer
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        chunk = tokens[start:start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks
```

**Sentence-aware chunking** — respect sentence boundaries, group until size limit:

```python
import re

def chunk_sentences(text: str, max_tokens: int = 400, overlap_sentences: int = 1) -> list[str]:
    """
    Better than fixed-size for prose. Chunks never split mid-sentence.
    Overlap_sentences: carry last N sentences into next chunk.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > max_tokens and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s.split()) for s in current)
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))
    return chunks
```

**Recursive / structural chunking** — respect document structure (headers, paragraphs, code blocks):

```python
def chunk_markdown(text: str, max_tokens: int = 512) -> list[str]:
    """
    Split on markdown headers first, then recurse if section is too large.
    Best for structured docs (READMEs, wikis, manuals).
    DSA inside: recursive splitting with size constraint.
    """
    # Split on headers
    sections = re.split(r'\n(?=#{1,3} )', text)
    chunks = []
    for section in sections:
        if len(section.split()) <= max_tokens:
            chunks.append(section.strip())
        else:
            # Recurse: split on paragraphs
            paragraphs = section.split('\n\n')
            buf, buf_len = [], 0
            for para in paragraphs:
                para_len = len(para.split())
                if buf_len + para_len > max_tokens and buf:
                    chunks.append('\n\n'.join(buf))
                    buf, buf_len = [], 0
                buf.append(para)
                buf_len += para_len
            if buf:
                chunks.append('\n\n'.join(buf))
    return [c for c in chunks if c.strip()]
```

**Semantic chunking** — split when cosine similarity between adjacent sentences drops below threshold:

```python
def chunk_semantic(sentences: list[str], embed_fn, threshold: float = 0.6) -> list[str]:
    """
    Group sentences with similar embedding into the same chunk.
    Split when meaning shifts significantly.
    More expensive (requires embedding at chunk time) but better quality.
    """
    if not sentences:
        return []
    embeddings = [embed_fn(s) for s in sentences]
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i-1], embeddings[i])
        if sim >= threshold:
            current.append(sentences[i])
        else:
            chunks.append(" ".join(current))
            current = [sentences[i]]
    if current:
        chunks.append(" ".join(current))
    return chunks
```

### Chunking tradeoffs

| Strategy | Pros | Cons | Best for |
|----------|------|------|----------|
| Fixed-size | Fast, simple | Splits mid-sentence | Uniform text, large corpora |
| Sentence-aware | Clean boundaries | Ignores structure | Prose, articles |
| Structural | Preserves hierarchy | Needs parsing | Markdown, HTML, PDFs |
| Semantic | Best quality | Slow, needs embeddings | High-value documents |

{: .note }
**The chunking default that works:** 512 tokens, 64-token overlap, sentence-aware splitting. Add structural splitting for structured docs. Add semantic splitting only if Recall@5 is below 0.7.

---

## 4. Embedding

### What an embedding is

An embedding model converts a chunk of text into a dense vector in ℝᵈ where d is typically 384–1536. Semantically similar texts land near each other in this space.

The embedding model is trained with a contrastive objective: pull similar pairs together, push dissimilar pairs apart. The most common training signal is (query, relevant passage) pairs from search logs or human annotation.

### Embedding model choices

| Model | Dimensions | Context | Best for |
|-------|-----------|---------|----------|
| `text-embedding-3-small` (OpenAI) | 1536 | 8191 tokens | General English |
| `text-embedding-3-large` (OpenAI) | 3072 | 8191 tokens | Highest quality general |
| `bge-large-en-v1.5` (BAAI) | 1024 | 512 tokens | Open-source, competitive |
| `e5-mistral-7b-instruct` | 4096 | 32k tokens | Long documents |
| `voyage-code-2` (Voyage) | 1536 | 16k tokens | Code retrieval |

**Key rule:** always use the same model for indexing and querying. Mixing models produces garbage similarity scores.

### Asymmetric vs. symmetric embedding

- **Symmetric:** query and passage use the same representation. Good for document similarity, deduplication.
- **Asymmetric:** query and passage go through different heads (or different prompts). Better for Q&A RAG — queries are short and interrogative, passages are long and declarative.

Most production RAG uses asymmetric embedding. BGE and E5 models support this via instruction prefixes:
```
Query: "Represent this sentence for searching relevant passages: {query}"
Passage: "Represent this passage for retrieval: {passage}"
```

### Batched embedding with rate limiting

```python
import time

def embed_corpus(
    chunks: list[str],
    embed_fn,           # (list[str]) → list[list[float]]
    batch_size: int = 100,
    rate_limit_rps: float = 10.0,
) -> list[list[float]]:
    """
    Embed a large corpus in batches with rate limiting.
    Always batch — single-item embedding is 10-50x slower than batched.
    """
    embeddings = []
    delay = 1.0 / rate_limit_rps
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_embeddings = embed_fn(batch)
        embeddings.extend(batch_embeddings)
        if i + batch_size < len(chunks):
            time.sleep(delay)
    return embeddings
```

---

## 5. Indexing

### Flat index (brute force)

The simplest index: store all vectors, compare query to every vector at query time.

```python
import math

class FlatIndex:
    """
    O(n × d) per query. Exact. Unacceptable above ~100k vectors.
    Use for: prototypes, small corpora (<50k chunks), eval baselines.
    """
    def __init__(self):
        self.vectors: list[list[float]] = []
        self.metadata: list[dict] = []

    def add(self, vector: list[float], meta: dict):
        self.vectors.append(vector)
        self.metadata.append(meta)

    def search(self, query: list[float], k: int = 5) -> list[tuple[dict, float]]:
        scores = [(cosine_similarity(query, v), i) for i, v in enumerate(self.vectors)]
        scores.sort(reverse=True)
        return [(self.metadata[i], s) for s, i in scores[:k]]
```

### HNSW (Hierarchical Navigable Small Worlds)

The dominant ANN algorithm used by every production vector DB. See [Ch 2]({% link docs/part1/chapter02-topk.md %}) for the intuition. Here's how the parameters affect production:

| Parameter | Effect | Typical value |
|-----------|--------|---------------|
| `M` (connections per node) | Higher → better recall, more memory | 16–64 |
| `ef_construction` | Higher → better index quality, slower build | 100–400 |
| `ef_search` | Higher → better recall, slower query | 50–200 |
| `space` | Cosine vs. L2 (use cosine for embeddings) | cosine |

```python
# Using hnswlib (pip install hnswlib)
import hnswlib
import numpy as np

class HNSWIndex:
    """
    Production-grade ANN index. O(log n) query, ~1-5% recall tradeoff.
    """
    def __init__(self, dim: int, max_elements: int, M: int = 32, ef_construction: int = 200):
        self.dim = dim
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
        self.metadata: dict[int, dict] = {}
        self._next_id = 0

    def add(self, vector: list[float], meta: dict):
        self.index.add_items(np.array([vector], dtype=np.float32), [self._next_id])
        self.metadata[self._next_id] = meta
        self._next_id += 1

    def add_batch(self, vectors: list[list[float]], metas: list[dict]):
        ids = list(range(self._next_id, self._next_id + len(vectors)))
        self.index.add_items(np.array(vectors, dtype=np.float32), ids)
        for i, meta in zip(ids, metas):
            self.metadata[i] = meta
        self._next_id += len(vectors)

    def search(self, query: list[float], k: int = 5, ef: int = 100) -> list[tuple[dict, float]]:
        self.index.set_ef(ef)
        labels, distances = self.index.knn_query(np.array([query], dtype=np.float32), k=k)
        results = []
        for label, dist in zip(labels[0], distances[0]):
            sim = 1 - dist   # hnswlib cosine returns distance, not similarity
            results.append((self.metadata[label], sim))
        return results
```

### Inverted index for sparse retrieval (BM25)

Vector search finds semantically similar chunks. But sometimes exact keyword match is better — especially for technical terms, product names, code identifiers. BM25 is the standard sparse retrieval algorithm.

```python
from collections import defaultdict
import math

class BM25Index:
    """
    Sparse keyword retrieval. Complements dense vector search.
    DSA inside: inverted index (HashMap of token → posting list).
    
    BM25 score:
        score(q, d) = Σ_t IDF(t) × (tf(t,d) × (k1+1)) / (tf(t,d) + k1×(1-b+b×|d|/avgdl))
    where:
        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))
        k1 = 1.5 (term frequency saturation)
        b  = 0.75 (length normalization)
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.index: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term → [(doc_id, tf)]
        self.doc_lens: list[int] = []
        self.docs: list[str] = []
        self.N = 0

    def add_documents(self, docs: list[str]):
        for doc_id, doc in enumerate(docs):
            tokens = doc.lower().split()
            self.doc_lens.append(len(tokens))
            self.docs.append(doc)
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for term, count in tf.items():
                self.index[term].append((doc_id, count))
        self.N = len(docs)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        avg_dl = sum(self.doc_lens) / self.N if self.N else 1
        query_terms = query.lower().split()
        scores: dict[int, float] = defaultdict(float)

        for term in query_terms:
            if term not in self.index:
                continue
            posting = self.index[term]
            df = len(posting)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            for doc_id, tf in posting:
                dl = self.doc_lens[doc_id]
                norm_tf = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / avg_dl))
                scores[doc_id] += idf * norm_tf

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.docs[doc_id], score) for doc_id, score in ranked[:k]]
```

---

## 6. Retrieval

### Dense retrieval

Standard ANN search over the vector index. The query is embedded, then K nearest chunks are returned by cosine similarity.

### Hybrid search: dense + sparse fusion

Dense retrieval excels at semantic similarity. Sparse retrieval (BM25) excels at exact keyword match. Combining them outperforms either alone.

**Reciprocal Rank Fusion (RRF)** — the standard fusion method:

```python
def reciprocal_rank_fusion(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60,
    top_n: int = 10
) -> list[tuple[str, float]]:
    """
    Combine multiple ranked lists without needing to normalize scores.
    
    RRF score: Σ_list 1 / (k + rank_in_list)
    
    k=60 is the standard default (from the original paper).
    Higher k reduces the importance of top-ranked documents.
    
    DSA inside: HashMap aggregation over ranked lists.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    for results in result_lists:
        for rank, (doc, _score) in enumerate(results, start=1):
            rrf_scores[doc] += 1.0 / (k + rank)
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


class HybridRetriever:
    def __init__(self, dense_index: HNSWIndex, sparse_index: BM25Index,
                 embed_fn, k: int = 20):
        self.dense = dense_index
        self.sparse = sparse_index
        self.embed_fn = embed_fn
        self.k = k

    def retrieve(self, query: str, top_n: int = 5) -> list[str]:
        # Dense retrieval
        q_emb = self.embed_fn(query)
        dense_results = [(meta['text'], score) for meta, score in self.dense.search(q_emb, k=self.k)]
        # Sparse retrieval
        sparse_results = self.sparse.search(query, k=self.k)
        # Fuse
        fused = reciprocal_rank_fusion([dense_results, sparse_results], top_n=top_n)
        return [doc for doc, _ in fused]
```

### Multi-query retrieval

A single query may not capture all relevant angles. Generate multiple query variants, retrieve for each, fuse results:

```python
def multi_query_retrieve(
    query: str,
    retriever,
    rephrase_fn,    # (query) → list[str] of rephrased queries
    k_per_query: int = 10,
    top_n: int = 5,
) -> list[str]:
    """
    Generate N query variants → retrieve K for each → fuse with RRF.
    Handles queries where a single embedding misses relevant angles.
    """
    queries = [query] + rephrase_fn(query)   # e.g. LLM generates 3 variants
    all_results = []
    seen = set()
    for q in queries:
        results = retriever.retrieve(q, top_n=k_per_query)
        ranked = [(doc, 1.0) for doc in results if doc not in seen]
        seen.update(doc for doc, _ in ranked)
        all_results.append(ranked)
    return [doc for doc, _ in reciprocal_rank_fusion(all_results, top_n=top_n)]
```

---

## 7. Reranking

Retrieval finds candidates fast but imprecisely. A reranker reads the query AND each candidate together (cross-attention) and produces a more accurate relevance score — at higher latency.

```
Retrieval:  O(log n), recall@20 ≈ 0.90, very fast
Reranking:  O(K × L), precision@5 much higher, slower (100-500ms for K=20)
```

### Cross-encoder reranker from scratch concept

```python
class CrossEncoderReranker:
    """
    A cross-encoder reads (query, passage) as a single input.
    This allows full attention between query and passage tokens —
    far more powerful than comparing independent embeddings.
    
    In production: use Cohere Rerank API, or
    open-source models like ms-marco-MiniLM-L-6-v2.
    
    Latency budget: ~50-200ms for K=20 passages, L=256 tokens each.
    """
    def __init__(self, score_fn):
        # score_fn: (query: str, passage: str) → float in [0, 1]
        self.score_fn = score_fn

    def rerank(self, query: str, passages: list[str], top_k: int = 5) -> list[str]:
        scored = [(p, self.score_fn(query, p)) for p in passages]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:top_k]]


def retrieve_and_rerank(
    query: str,
    retriever,
    reranker: CrossEncoderReranker,
    retrieve_k: int = 20,
    rerank_top_k: int = 5,
) -> list[str]:
    """
    Two-stage retrieval: fast ANN → slow but accurate reranker.
    Standard production pattern for high-quality RAG.
    """
    candidates = retriever.retrieve(query, top_n=retrieve_k)
    return reranker.rerank(query, candidates, top_k=rerank_top_k)
```

---

## 8. Prompt building

The retrieved chunks must be packed into the LLM's context window efficiently.

```python
def build_rag_prompt(
    query: str,
    chunks: list[str],
    max_context_tokens: int = 3000,
    system_prompt: str = "Answer the question using only the provided context. If the answer is not in the context, say so.",
) -> str:
    """
    Pack retrieved chunks into context window.
    
    Key decisions:
    1. How many chunks to include (budget-constrained)
    2. Chunk ordering (most relevant first vs. "lost in the middle" awareness)
    3. Citation format (numbered vs. inline)
    4. What to say when context is insufficient
    """
    # "Lost in the middle" effect: LLMs attend better to start and end of context.
    # Mitigate by placing highest-relevance chunks first AND last.
    context_parts = []
    total_tokens = 0
    token_budget = max_context_tokens

    for i, chunk in enumerate(chunks):
        chunk_tokens = len(chunk.split())  # rough estimate; use tokenizer in production
        if total_tokens + chunk_tokens > token_budget:
            break
        context_parts.append(f"[{i+1}] {chunk}")
        total_tokens += chunk_tokens

    context = "\n\n".join(context_parts)

    return f"""{system_prompt}

Context:
{context}

Question: {query}
Answer:"""
```

### Context window management

| Model | Context limit | Usable for RAG context |
|-------|--------------|----------------------|
| GPT-4o | 128k tokens | ~100k tokens (leave room for output) |
| Claude Sonnet | 200k tokens | ~150k tokens |
| Llama 3 70B | 8k tokens | ~6k tokens |
| Gemini 1.5 Pro | 1M tokens | ~800k tokens |

Even with large context windows, stuffing 100k tokens of context degrades generation quality. Sweet spot for most RAG systems: 2k–8k tokens of context (5–20 chunks of 200–400 tokens each).

---

## 9. Evaluation

Evaluating RAG requires measuring two things separately: **retrieval quality** and **generation quality**.

### Retrieval metrics

```python
def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of relevant docs found in top-K. Primary retrieval metric."""
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of top-K results that are relevant."""
    return len(set(retrieved[:k]) & relevant) / k if k else 0.0


def mean_reciprocal_rank(relevant: set[str], retrieved: list[str]) -> float:
    """
    MRR: 1/rank of first relevant result. Good for single-answer queries.
    MRR=1.0 means the first result is always relevant.
    MRR=0.5 means the first relevant result is at rank 2 on average.
    """
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """
    NDCG: normalized discounted cumulative gain.
    Rewards finding relevant docs earlier. Standard ranking metric.
    DSA inside: logarithmic discount + normalization.
    """
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(retrieved[:k], start=1)
        if doc in relevant
    )
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_retrieval(
    eval_set: list[dict],   # [{"query": ..., "relevant_ids": [...]}]
    retriever,
    k: int = 5,
) -> dict:
    """Run full retrieval evaluation over a labeled eval set."""
    recalls, precisions, mrrs, ndcgs = [], [], [], []
    for item in eval_set:
        relevant = set(item["relevant_ids"])
        retrieved = retriever.retrieve(item["query"], top_n=k * 2)
        retrieved_ids = [r["id"] for r in retrieved]
        recalls.append(recall_at_k(relevant, retrieved_ids, k))
        precisions.append(precision_at_k(relevant, retrieved_ids, k))
        mrrs.append(mean_reciprocal_rank(relevant, retrieved_ids))
        ndcgs.append(ndcg_at_k(relevant, retrieved_ids, k))
    return {
        f"Recall@{k}": sum(recalls) / len(recalls),
        f"Precision@{k}": sum(precisions) / len(precisions),
        "MRR": sum(mrrs) / len(mrrs),
        f"NDCG@{k}": sum(ndcgs) / len(ndcgs),
    }
```

### Generation metrics

```python
def answer_faithfulness(answer: str, context_chunks: list[str], score_fn) -> float:
    """
    Does every claim in the answer appear in the retrieved context?
    Measures hallucination rate.
    score_fn: (claim, context) → float (use an LLM judge in production)
    """
    # Split answer into claims (sentences as proxy)
    claims = [s.strip() for s in answer.split('.') if s.strip()]
    context = " ".join(context_chunks)
    scores = [score_fn(claim, context) for claim in claims]
    return sum(scores) / len(scores) if scores else 0.0


def answer_relevance(query: str, answer: str, score_fn) -> float:
    """Does the answer actually address the question? Score 0-1."""
    return score_fn(query, answer)
```

### The RAGAS framework (production standard)

RAGAS (Retrieval-Augmented Generation Assessment) defines four metrics evaluated by an LLM judge:

| Metric | Measures | Formula |
|--------|----------|---------|
| **Faithfulness** | Is answer grounded in context? | claims supported / total claims |
| **Answer Relevance** | Does answer address the question? | LLM score |
| **Context Precision** | Are retrieved chunks actually relevant? | relevant retrieved / total retrieved |
| **Context Recall** | Are all relevant chunks retrieved? | relevant retrieved / total relevant |

A good RAG system targets: Faithfulness > 0.85, Context Recall > 0.80.

---

## 10. Production architecture

### Component choices by scale

| Scale | Vector DB | Embedding | Reranker |
|-------|-----------|-----------|---------|
| Prototype (<100k chunks) | In-memory HNSW (hnswlib) | OpenAI ada | None |
| Small production (<10M chunks) | Chroma, Weaviate, Qdrant | OpenAI or BGE | Cohere or MiniLM |
| Large production (>100M chunks) | Pinecone, Milvus, Weaviate | Fine-tuned model | Fine-tuned cross-encoder |

### Full production RAG system

```python
class ProductionRAG:
    """
    Production-grade RAG with hybrid retrieval, reranking, and evaluation.
    """
    def __init__(
        self,
        vector_index: HNSWIndex,
        bm25_index: BM25Index,
        embed_fn,
        reranker: CrossEncoderReranker,
        llm_fn,             # (prompt: str) → str
        retrieve_k: int = 20,
        rerank_k: int = 5,
        max_context_tokens: int = 3000,
    ):
        self.retriever = HybridRetriever(vector_index, bm25_index, embed_fn, k=retrieve_k)
        self.reranker = reranker
        self.llm = llm_fn
        self.rerank_k = rerank_k
        self.max_context_tokens = max_context_tokens

    def answer(self, query: str) -> dict:
        # Stage 1: Retrieve candidates
        candidates = self.retriever.retrieve(query, top_n=20)
        # Stage 2: Rerank
        top_chunks = self.reranker.rerank(query, candidates, top_k=self.rerank_k)
        # Stage 3: Build prompt
        prompt = build_rag_prompt(query, top_chunks, self.max_context_tokens)
        # Stage 4: Generate
        answer = self.llm(prompt)
        return {
            "answer": answer,
            "retrieved_chunks": top_chunks,
            "prompt_tokens": len(prompt.split()),
        }
```

### Common failure modes and fixes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Wrong chunk size | Answers miss key details | Tune chunk size; try 256, 512, 1024 tokens |
| Embedding mismatch | Low Recall@K on eval set | Use same model for index and query |
| Missing exact terms | Technical queries fail | Add BM25 to hybrid retrieval |
| Context noise | Model ignores context | Add reranker; reduce K |
| Lost in the middle | Middle chunks ignored | Place most relevant chunks first and last |
| Hallucination despite retrieval | High context recall, low faithfulness | Tighten system prompt; add citation instructions |
| Stale index | Answers reflect old docs | Implement incremental indexing with doc versioning |

---

## 11. Interview questions

**System design:** Design a RAG system for a 10M-document enterprise knowledge base. How would you handle document updates? What metrics would you monitor in production?

**Depth questions:**
- Why is cosine similarity preferred over dot product for retrieval? (It's length-normalized — a long document shouldn't automatically score higher.)
- What is the "lost in the middle" effect and how do you mitigate it?
- When would you use sparse retrieval (BM25) instead of dense retrieval?
- What does Recall@K measure and what's a reasonable target for a production RAG system?
- Why does chunk overlap improve retrieval quality?
- How would you update a vector index when a source document changes?

**Debugging:** Your RAG system has high retrieval recall but users still complain answers are wrong. Where do you look first? (Faithfulness — the model is ignoring the context. Check: is context window being exceeded? Is the system prompt strong enough? Is the reranker introducing noise?)

---

{: .highlight }
**Next:** [ML Spine Ch B — LLM Internals →]({% link docs/ml-spine/chapter-llm.md %})
