---
layout: default
title: "Ch 1 · Hashing, Token Counting, and LLM Context"
parent: "Part 1 — Algorithm Spine"
nav_order: 1
description: "HashMap from scratch → tokenizers → KV cache → agent tool registry."
---

# Hashing, Token Counting, and LLM Context
{: .no_toc }

*Chapter 1 · Part 1 — Algorithm Spine*
{: .text-grey-dk-000 .fs-4 }

<div class="reading-meta">
  ⏱ 25 min read &nbsp;·&nbsp;
  🟦 Engineering: L3 &nbsp;·&nbsp;
  🟩 ML: L2–L3
</div>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dpaul0501/learning-systems-from-scratch/blob/main/notebooks/chapter01_hashing.ipynb)
&nbsp;
[![View Notebook](https://img.shields.io/badge/Notebook-View%20on%20GitHub-lightgrey?logo=github)](https://github.com/dpaul0501/learning-systems-from-scratch/blob/main/notebooks/chapter01_hashing.ipynb)

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Why this topic matters

HashMap is the most frequently used data structure in real software — and in real ML systems.

Every LeetCode medium problem has a HashMap hiding inside it.  
Every tokenizer is a HashMap.  
Every LLM's KV cache is a HashMap with eviction logic.  
Every agent's tool registry is a HashMap.  
Every feature store lookup is a HashMap at its core.

| Layer | What it is | HashMap role |
|-------|-----------|--------------|
| Tokenizer | Vocabulary | `token_string → token_id` |
| Embedding table | Lookup | `token_id → float[d_model]` |
| KV Cache | Reuse attention | `(layer, head, position) → (K, V)` |
| Feature store | Fast feature retrieval | `entity_id → feature_vector` |
| Agent tool registry | Tool dispatch | `tool_name → function` |

---

## 2. Engineering view

### The core abstraction

A hash map maps a **key** to a **value** in O(1) average time by:
1. Hashing the key to an integer (the hash function)
2. Using the integer as an index into an array (the bucket array)
3. Handling collisions when two keys hash to the same bucket

### Collision resolution

**Chaining:** Each bucket holds a linked list. Worst case O(n) if all keys collide.  
**Open addressing:** On collision, probe to the next open slot. Used in Python's `dict`.

Python dicts use open addressing with a smart probe sequence. Load factor is kept below ~66% and the table doubles when full — keeping average O(1).

### Core operations

| Operation | Average | Worst |
|-----------|---------|-------|
| Insert | O(1) | O(n) |
| Lookup | O(1) | O(n) |
| Delete | O(1) | O(n) |
| Space | O(n) | O(n) |

---

## 3. Core intuition

{: .highlight }
Counting is just building a frequency map.  
Deduplication is just building a set.  
Grouping is just inverting a map.  
Caching is just a map with eviction.

Every "hard" HashMap problem is one of these four operations applied non-obviously.

---

## 4. From-scratch implementations

### HashMap from scratch

```python
class HashMap:
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.size = 0
        self.buckets = [[] for _ in range(capacity)]
        self.load_factor_limit = 0.67

    def _hash(self, key):
        return hash(key) % self.capacity

    def put(self, key, value):
        idx = self._hash(key)
        bucket = self.buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self.size += 1
        if self.size / self.capacity > self.load_factor_limit:
            self._resize()

    def get(self, key, default=None):
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        return default

    def _resize(self):
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        for bucket in old_buckets:
            for k, v in bucket:
                self.put(k, v)
```

### LRU Cache — the production caching primitive

The LRU Cache is the most important HashMap variant in ML systems. Used in model serving, tokenizer caching, and KV cache management.

```python
from collections import OrderedDict

class LRUCache:
    """O(1) get and put using HashMap + doubly linked list."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

### Tokenizer vocabulary — what tokenizers actually are

```python
class Vocabulary:
    """A tokenizer is a HashMap from string → int and int → string."""
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self._next_id = 0
        for special in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            self.add(special)

    def add(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = self._next_id
            self.id_to_token[self._next_id] = token
            self._next_id += 1
        return self.token_to_id[token]

    def encode(self, text: str) -> list[int]:
        return [
            self.token_to_id.get(t, self.token_to_id["<UNK>"])
            for t in text.split()
        ]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id_to_token.get(i, "<UNK>") for i in ids)
```

### Feature hashing — the hashing trick

```python
import hashlib

class FeatureHasher:
    """Maps arbitrary string features to a fixed-size vector. No vocabulary needed."""
    def __init__(self, n_features: int = 2**18):
        self.n_features = n_features

    def _hash(self, feature: str) -> int:
        h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
        return h % self.n_features

    def transform(self, features: list[str]) -> dict[int, float]:
        vec = {}
        for f in features:
            idx = self._hash(f)
            vec[idx] = vec.get(idx, 0) + 1.0
        return vec
```

### Agent tool registry

```python
class ToolRegistry:
    """An agent's tool registry is a HashMap from name → callable."""
    def __init__(self):
        self._tools = {}
        self._schemas = {}

    def register(self, name: str, fn, schema: dict):
        self._tools[name] = fn
        self._schemas[name] = schema

    def call(self, name: str, args: dict):
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}. Available: {list(self._tools)}")
        return self._tools[name](**args)
```

---

## 5. KV cache in LLMs — the scaling consequence

The KV cache stores Key and Value tensors so the model doesn't recompute attention for past tokens on each generation step.

```
Without KV cache: O(n²) attention per token generated
With KV cache:    O(n)  attention per token generated
```

Memory cost: `2 × n_layers × seq_len × d_head × n_heads × bytes_per_float`

For a 70B model with 4096 seq length: ~35GB just for KV cache. This is why KV cache eviction (LRU or sliding window) is a real system design problem — and why you'll be asked about it in MLE system design rounds.

---

## 6. Interview patterns

| You see... | You think... |
|-----------|-------------|
| "find pair with sum X" | complement map |
| "count frequencies" | `Counter` / frequency map |
| "group by property" | invert the map |
| "find duplicates" | seen set |
| "cache recent results" | LRU Cache |
| "O(1) lookup on dynamic set" | HashMap |

---

## 7. Practice problems

### LeetCode

| # | Problem | Pattern |
|---|---------|---------|
| 1 | [Two Sum](https://leetcode.com/problems/two-sum/) | Complement map |
| 49 | [Group Anagrams](https://leetcode.com/problems/group-anagrams/) | Invert map |
| 347 | [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) | Frequency + heap |
| 128 | [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/) | Set membership |
| 146 | [LRU Cache](https://leetcode.com/problems/lru-cache/) | HashMap + deque |
| 560 | [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/) | Prefix sum map |

### ML implementation problems

1. Build a bag-of-words vectorizer from scratch (no sklearn)
2. Implement LRU cache for an embedding lookup service with TTL
3. Implement the feature hashing trick for a streaming predictor
4. Build a simple tool registry for a ReAct-style agent with retry logic

### ML theory questions

{: .interview-q }
Why does a tokenizer need both `token_to_id` and `id_to_token`?

{: .interview-q }
Why does the KV cache make LLM generation O(n) instead of O(n²)?

{: .interview-q }
What is the hashing trick and when does it cause problems?

{: .interview-q }
How does Python's `dict` avoid worst-case collisions at startup?

### System design

{: .interview-q }
Design a feature store that serves features at <10ms p99 latency.

{: .interview-q }
Design the caching layer for an LLM inference service serving 10K QPS.

---

## 8. Advanced rabbit holes

**Bloom filters** — probabilistic set membership in O(1) with zero per-item memory. Used for deduplication in LLM pretraining, URL dedup in crawlers, feature lookup validation.

**Count-Min Sketch** — probabilistic frequency counter. Returns an overestimate, never an underestimate. Used in streaming word counts, top-K heavy hitters in telemetry.

**HyperLogLog** — estimates unique cardinality using O(log log n) space. Redis's `PFCOUNT`. Used in counting unique users, unique tokens in a corpus.

**Consistent hashing** — when a HashMap is distributed across N servers, adding or removing a server shouldn't invalidate all keys. Used in Cassandra, Redis Cluster, CDNs.

---

## 9. Route guidance

**Engineering-inclined (target L3):** Implement HashMap from scratch. Solve all 6 LeetCode problems. Understand LRU Cache deeply — it appears in system design constantly. Read the KV cache section carefully.

**ML-inclined (target L2–L3):** Solve Two Sum, Group Anagrams, LRU Cache. Read the tokenizer and KV cache implementations — they demystify tools you use daily. The feature hashing section is directly applicable to your work.

---

{: .highlight }
**Next:** [Chapter 2 — Top-K, Vector Search, and RAG Retrieval →]({% link docs/part1/chapter02-topk.md %})
