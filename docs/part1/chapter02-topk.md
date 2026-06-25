---
layout: default
title: "Ch 2 · Top-K, Vector Search, and RAG Retrieval"
parent: "Part 1 — Algorithm Spine"
nav_order: 2
description: "MinHeap from scratch → KNN → brute-force vector search → Mini RAG → beam search."
---

# Top-K, Vector Search, and RAG Retrieval
{: .no_toc }

*Chapter 2 · Part 1 — Algorithm Spine*
{: .text-grey-dk-000 .fs-4 }

<div class="reading-meta">
  ⏱ 30 min read &nbsp;·&nbsp;
  🟦 Engineering: L3 &nbsp;·&nbsp;
  🟩 ML: L3
</div>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mldsa/mldsa.github.io/blob/main/notebooks/chapter02_topk.ipynb)
&nbsp;
[![View Notebook](https://img.shields.io/badge/Notebook-View%20on%20GitHub-lightgrey?logo=github)](https://github.com/mldsa/mldsa.github.io/blob/main/notebooks/chapter02_topk.ipynb)

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Why this topic matters

Ask a senior ML engineer what they built last quarter. The answer is often a retrieval system.

RAG (Retrieval-Augmented Generation) is now the dominant pattern for grounding LLMs in real data. Every RAG system has one core operation at its heart: **find the K most relevant items from a large set.**

This is the Top-K problem.

| System | What "K" means | Data structure |
|--------|---------------|----------------|
| KNN | K nearest labeled examples | Heap or sorted list |
| Candidate generation | K candidate items per user | Approximate heap |
| Beam search | K most likely token sequences | Priority queue |
| Vector search | K most similar embeddings | HNSW / IVF index |
| RAG retrieval | K most relevant document chunks | Vector DB |
| Reranking | Top K after initial retrieval | Sorted list |
| Monitoring | K most frequent error types | Count-Min + heap |

**For the engineering-inclined reader:** Heaps are in 15+ interview problems. Top-K retrieval is a core system design question at every ML company.

**For the ML-inclined reader:** You call `retriever.get_relevant_documents(query)` and trust the vector DB. This chapter shows you exactly what happens inside — and what breaks at scale.

---

## 2. Engineering view

### The heap

A **binary heap** is a complete binary tree where every parent satisfies the heap property:
- **Min-heap**: parent ≤ children. Root = minimum element.
- **Max-heap**: parent ≥ children. Root = maximum element.

Stored as an array. For node at index `i`:
- Left child: `2i + 1`
- Right child: `2i + 2`
- Parent: `(i - 1) // 2`

### Core operations

| Operation | Time | How |
|-----------|------|-----|
| Push (insert) | O(log n) | Add to end, bubble up |
| Pop (extract min/max) | O(log n) | Swap root with last, bubble down |
| Peek (see min/max) | O(1) | Read root |
| Heapify (build from list) | O(n) | Not O(n log n) — math works out |
| Top-K from n elements | O(n log k) | Maintain size-k heap |

**Why heap for Top-K instead of sorting?**
Sort is O(n log n). A heap of size K gives O(n log k). When k << n (K=10 from 1M items), this is nearly O(n).

### Quickselect

For finding the Kth largest element (not sorted Top-K), Quickselect runs in O(n) average — faster than a heap. But it doesn't give sorted order and has O(n²) worst case.

- Need sorted Top-K list → heap: O(n log k)
- Need just Kth element → Quickselect: O(n) average
- Streaming Top-K (n unknown) → always use heap

---

## 3. ML / LLM / RAG view

The Top-K pattern appears everywhere in modern ML. The same primitive — maintain the K best items — scales from 100 items (use a heap) to 1 billion items (use an approximate index like HNSW).

**The heap insight:**

> Instead of sorting everything and taking the top K, maintain a window of size K and evict the worst item whenever something better arrives.

**The vector search insight:**

> Semantic similarity is a geometric problem. Similar meaning → nearby points in embedding space. Top-K retrieval → nearest neighbor search.

The leap from heap → RAG is: instead of comparing numbers, we compare vectors with cosine similarity. The Top-K problem is the same; the distance function changes.

---

## 4. Mathematical foundation

### Cosine similarity

For vectors **u** and **v**:

```
cosine_similarity(u, v) = (u · v) / (||u|| × ||v||)
```

Range: [-1, 1]. Value of 1 = identical direction. Value of 0 = orthogonal (unrelated).

Why cosine over Euclidean for embeddings? Embeddings encode direction (meaning), not magnitude. A short sentence and a long sentence about the same topic should be similar — cosine handles this, Euclidean does not.

### The recall-latency tradeoff

Exact nearest neighbor search requires comparing every query to every stored vector: **O(n × d)** per query, where d is the embedding dimension.

For n=10M, d=1536 (OpenAI ada-002): 15.36 billion float multiplications per query. At 1ns per op: **15 seconds per query.** Unacceptable.

Approximate nearest neighbor (ANN) algorithms trade **recall** (fraction of true nearest neighbors returned) for **latency**:

```
Exact search:  recall=1.00, latency=15s
HNSW (tuned):  recall=0.95, latency=2ms
```

For RAG, missing 5% of relevant chunks rarely hurts answer quality. The latency saving is worth it.

---

## 5. From-scratch implementations

### 5.1 Min-heap from scratch

```python
class MinHeap:
    """
    A binary min-heap. Parent is always ≤ children.
    Used for Top-K largest (counterintuitive but correct).
    """
    def __init__(self):
        self.data = []

    def push(self, val):
        self.data.append(val)
        self._bubble_up(len(self.data) - 1)

    def pop(self):
        if len(self.data) == 1:
            return self.data.pop()
        root = self.data[0]
        self.data[0] = self.data.pop()
        self._bubble_down(0)
        return root

    def peek(self):
        return self.data[0]

    def __len__(self):
        return len(self.data)

    def _bubble_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.data[parent] > self.data[i]:
                self.data[parent], self.data[i] = self.data[i], self.data[parent]
                i = parent
            else:
                break

    def _bubble_down(self, i):
        n = len(self.data)
        while True:
            smallest = i
            left, right = 2 * i + 1, 2 * i + 2
            if left < n and self.data[left] < self.data[smallest]:
                smallest = left
            if right < n and self.data[right] < self.data[smallest]:
                smallest = right
            if smallest == i:
                break
            self.data[i], self.data[smallest] = self.data[smallest], self.data[i]
            i = smallest
```

### 5.2 Top-K frequent elements (LeetCode #347)

```python
import heapq
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """
    O(n log k) — count frequencies, then heap.
    ML analog: find top-K tokens in a corpus,
    top-K features by importance, top-K errors by frequency.
    """
    freq = Counter(nums)
    heap = []
    for elem, count in freq.items():
        heapq.heappush(heap, (count, elem))
        if len(heap) > k:
            heapq.heappop(heap)
    return [elem for _, elem in heap]

print(top_k_frequent([1,1,1,2,2,3], k=2))  # [1, 2]
```

### 5.3 KNN classifier from scratch

```python
import heapq
from collections import Counter

class KNNClassifier:
    """
    K-Nearest Neighbors. Top-K retrieval over a labeled dataset.
    Exact same structure as K Closest Points, applied to feature
    vectors with class labels.
    """
    def __init__(self, k: int = 3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean(self, a, b):
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5

    def predict_one(self, x):
        heap = []
        for i, x_train in enumerate(self.X_train):
            dist = self._euclidean(x, x_train)
            heapq.heappush(heap, (-dist, self.y_train[i]))
            if len(heap) > self.k:
                heapq.heappop(heap)
        labels = [label for _, label in heap]
        return Counter(labels).most_common(1)[0][0]

    def predict(self, X):
        return [self.predict_one(x) for x in X]

clf = KNNClassifier(k=3)
clf.fit([[1,1], [2,2], [3,3], [6,6], [7,7]], [0, 0, 0, 1, 1])
print(clf.predict([[2.5, 2.5], [6.5, 6.5]]))  # [0, 1]
```

### 5.4 Brute-force vector search

```python
import heapq, math

def cosine_similarity(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai**2 for ai in a))
    norm_b = math.sqrt(sum(bi**2 for bi in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class BruteForceVectorSearch:
    """
    Exact nearest neighbor search by cosine similarity.
    O(n × d) per query — works for small n, unacceptable for n > 100k.
    This is what every vector DB does before it gets fast.
    """
    def __init__(self):
        self.vectors = []

    def add(self, doc_id: str, embedding: list):
        self.vectors.append((doc_id, embedding))

    def search(self, query: list, k: int = 5):
        heap = []
        for doc_id, emb in self.vectors:
            sim = cosine_similarity(query, emb)
            heapq.heappush(heap, (sim, doc_id))
            if len(heap) > k:
                heapq.heappop(heap)
        return sorted([(doc_id, sim) for sim, doc_id in heap], reverse=True)

index = BruteForceVectorSearch()
index.add("doc1", [0.9, 0.1, 0.0])
index.add("doc2", [0.8, 0.2, 0.1])
index.add("doc3", [0.1, 0.9, 0.0])
results = index.search([0.85, 0.15, 0.0], k=2)
print(results)  # [("doc1", ~0.99), ("doc2", ~0.97)]
```

### 5.5 Mini RAG system

```python
class MiniRAG:
    """
    RAG = Vector Search + Prompt Construction + LLM Generation.
    This implements the retrieval half from scratch.
    """
    def __init__(self, embed_fn, generate_fn, k: int = 3):
        self.embed_fn = embed_fn        # str → list[float]
        self.generate_fn = generate_fn  # (prompt) → str
        self.k = k
        self.index = BruteForceVectorSearch()
        self.docs = {}

    def add_document(self, doc_id: str, text: str):
        self.index.add(doc_id, self.embed_fn(text))
        self.docs[doc_id] = text

    def retrieve(self, query: str):
        results = self.index.search(self.embed_fn(query), k=self.k)
        return [self.docs[doc_id] for doc_id, _ in results]

    def answer(self, query: str) -> str:
        context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(self.retrieve(query)))
        return self.generate_fn(f"Answer using only the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:")

# At scale: BruteForceVectorSearch → HNSW index (Chroma, Pinecone, Weaviate)
```

### 5.6 Median from data stream (two heaps)

```python
import heapq

class MedianFinder:
    """
    Maintain two heaps: max_heap (lower half) + min_heap (upper half).
    ML use: online monitoring of latency, score distributions,
    feature value medians in streaming pipelines.
    """
    def __init__(self):
        self.max_heap = []  # lower half, negated
        self.min_heap = []  # upper half

    def add_num(self, num: int):
        heapq.heappush(self.max_heap, -num)
        if self.max_heap and self.min_heap and (-self.max_heap[0]) > self.min_heap[0]:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def find_median(self) -> float:
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        return (-self.max_heap[0] + self.min_heap[0]) / 2.0
```

---

## 6. Complexity analysis

| Task | Brute Force | Optimized | Structure |
|------|------------|-----------|-----------|
| Top-K from array | O(n log n) sort | O(n log k) | Min-heap of size k |
| Kth largest | O(n log n) | O(n) avg | Quickselect |
| KNN, n=1M, d=768 | O(n·d) = 768M ops | O(d log n) | ANN (HNSW) |
| Streaming Top-K | O(n log n) per update | O(log k) per item | Min-heap |
| Beam search, beam=5 | O(V^len) | O(len × V × k) | Heap per step |

---

## 7. Production / scaling concerns

For n=100M vectors, d=1536: brute-force = **153 seconds per query.** Real vector DBs use **HNSW** (Hierarchical Navigable Small Worlds) — O(log n) query time, 1000x latency improvement, ~5% recall tradeoff.

### Recall@K — the key RAG metric

```python
def recall_at_k(relevant: set, retrieved: list, k: int) -> float:
    """Fraction of relevant documents found in top-K results."""
    return len(set(retrieved[:k]) & relevant) / len(relevant)

relevant_docs = {"doc1", "doc3", "doc7"}
retrieved_docs = ["doc1", "doc2", "doc3", "doc5", "doc9"]
print(recall_at_k(relevant_docs, retrieved_docs, k=5))  # 0.667
```

---

## 8. Interview patterns

| You see... | You think... |
|-----------|-------------|
| "K largest/smallest" | Min-heap of size K |
| "K closest" | Max-heap of size K (negate distance) |
| "K most frequent" | Count + heap |
| "streaming K best" | Always heap, don't sort |
| "median from stream" | Two heaps (max + min) |
| "merge K sorted lists" | Heap with (value, list_index, element_index) |
| "K nearest in vector space" | Brute force if small, ANN if large |
| "RAG retrieval at scale" | HNSW / IVF index |

---

## 9. Practice problems

| # | Problem | Pattern | Difficulty |
|---|---------|---------|-----------|
| 215 | Kth Largest Element | Heap / Quickselect | Medium |
| 347 | Top K Frequent Elements | Count + heap | Medium |
| 973 | K Closest Points to Origin | Max-heap | Medium |
| 295 | Find Median from Data Stream | Two heaps | Hard |
| 23 | Merge K Sorted Lists | Min-heap | Hard |
| 378 | Kth Smallest in Matrix | Min-heap | Medium |
| 692 | Top K Frequent Words | Count + heap | Medium |

**ML problems:** Implement brute-force vector search and plot latency vs n. Build Recall@K and MRR evaluation functions. Build a mini RAG system over 20 text chunks using any embedding API.

**Theory questions:** Why does cosine similarity work better than Euclidean for embeddings? What is the curse of dimensionality at d=768? What does HNSW trade away to achieve O(log n) search?

---

## 10. Common mistakes

**LeetCode:** Using max-heap for Top-K largest (use min-heap of size K). Forgetting to negate values for max-heap in Python. Sorting the whole array when only K elements are needed.

**ML systems:** Using brute-force beyond ~50k vectors. Optimizing Recall@1 when Recall@5 is what matters for RAG. Not chunking documents properly. Embedding query and documents with different models.

---

## 11. Advanced: HNSW

HNSW (Hierarchical Navigable Small Worlds) is the dominant ANN algorithm used by Pinecone, Weaviate, Chroma, and others. It builds a multi-layer graph at index time, then navigates it greedily at query time:

```
Layer 2 (sparse):  1 ——— 4
Layer 1 (medium):  1 - 2 - 4 - 6
Layer 0 (dense):   1-2-3-4-5-6-7-8-9

Search: start at Layer 2 → greedy walk → drop to Layer 1 → refine → drop to Layer 0 → top-K
```

Time: O(log n) query, O(n log n) build. The 95% recall for 2ms latency tradeoff that makes RAG at scale practical.

---

{: .highlight }
**Previous:** [Ch 1 — Hashing, Token Counting, and LLM Context →]({% link docs/part1/chapter01-hashing.md %})
&nbsp;·&nbsp;
**Next:** [Ch 3 — Dynamic Programming, Beam Search, Viterbi →]({% link docs/part1/chapter03-dp.md %})
