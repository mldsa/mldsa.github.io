---
layout: default
title: "ML → DSA Index"
nav_order: 3
description: "Start from ML concepts. Find the algorithm powering every system you use."
---

# ML → DSA Index
{: .no_toc }

*For ML practitioners: every model and system you work with runs on an algorithm. Here's which one.*
{: .fs-5 .text-grey-dk-000 }

---

Use this index if you think in ML concepts first. Each row maps something you use daily to the data structure or algorithm powering it — and the chapter that implements it from scratch.

---

## Tokenization and Language Modeling

| ML Concept | Algorithm Inside | Chapter | Depth |
|-----------|-----------------|---------|-------|
| Tokenizer vocabulary | Bidirectional HashMap (`token↔id`) | [Ch 1]({% link docs/part1/chapter01-hashing.md %}) | L2 |
| BPE tokenization | Word Break DP · Greedy merge | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L3 |
| Trie-based tokenizer | Trie (prefix tree) | Ch 4 (coming) | L3 |
| N-gram language model | HashMap frequency counting | [Ch 1]({% link docs/part1/chapter01-hashing.md %}) | L2 |
| Perplexity computation | Log-probability accumulation · DP | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L2 |
| OOV / unknown token handling | HashMap with default fallback | [Ch 1]({% link docs/part1/chapter01-hashing.md %}) | L1 |

---

## LLM Generation and Decoding

| ML Concept | Algorithm Inside | Chapter | Depth |
|-----------|-----------------|---------|-------|
| Greedy decoding | Argmax over logits → O(V) scan | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L2 |
| Beam search | DP on sequences · Min-heap of size K | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L4 |
| Top-k sampling | Heap top-K + sampling | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L3 |
| Top-p (nucleus) sampling | Sorted cumulative sum · Binary search | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L3 |
| KV cache | LRU HashMap · Sliding window eviction | [Ch 1]({% link docs/part1/chapter01-hashing.md %}) | L4 |
| Speculative decoding | Dual DP (draft + target model) | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L4 |
| Constrained / structured output | Word Break DP · Trie + beam | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L4 |
| Temperature scaling | Softmax with divisor · numerical stability | Ch 5 (coming) | L2 |

---

## Retrieval, RAG, and Vector Search

| ML Concept | Algorithm Inside | Chapter | Depth |
|-----------|-----------------|---------|-------|
| Cosine similarity | Dot product · L2 norm | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L2 |
| KNN search (brute force) | Heap top-K · O(n × d) | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L2 |
| Approximate KNN (HNSW) | Multi-layer graph · greedy walk | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L4 |
| BM25 retrieval | Inverted index · TF-IDF HashMap | Ch 9 (coming) | L3 |
| Hybrid search | Merge K sorted lists · score fusion | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L3 |
| Reranker | Top-K from retrieval → reorder | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L3 |
| Recall@K | Heap + set intersection | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L2 |
| RAG chunking strategy | Interval DP · greedy segmentation | Ch 7 (coming) | L3 |
| Embedding deduplication | MinHash · LSH · Bloom filter | Ch 10 (coming) | L4 |

---

## Sequence Labeling and Structured Prediction

| ML Concept | Algorithm Inside | Chapter | Depth |
|-----------|-----------------|---------|-------|
| Named Entity Recognition (NER) | Viterbi DP on label sequences | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L4 |
| POS tagging | Viterbi · HMM decoding | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L4 |
| CRF decoding | Forward-backward DP + Viterbi | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L4 |
| Dynamic time warping | 2D DP alignment | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L3 |
| Sequence-to-sequence alignment | Edit distance · LCS | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L3 |

---

## Classical ML Algorithms

| ML Concept | Algorithm Inside | Chapter | Depth |
|-----------|-----------------|---------|-------|
| KNN classifier | Heap top-K · Euclidean distance | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L2 |
| Decision tree (Gini split) | Greedy search over feature thresholds | Ch 4 (coming) | L3 |
| Random forest | Bootstrap sampling · Parallel tree recursion | Ch 4 (coming) | L3 |
| Gradient boosting | Greedy additive model · Taylor DP | Ch 4 (coming) | L4 |
| XGBoost split finding | Histogram algorithm · sorted scan | Ch 4 (coming) | L4 |
| K-Means | Greedy assignment · centroid update | Ch 8 (coming) | L2 |
| PCA | SVD · matrix factorization | Ch 8 (coming) | L3 |

---

## Evaluation Metrics

| ML Metric | Algorithm Inside | Chapter | Depth |
|-----------|-----------------|---------|-------|
| ROUGE-L | Longest Common Subsequence (LCS) | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L2 |
| WER (Word Error Rate) | Edit distance at word level | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L2 |
| NDCG | Sorted ranking · logarithmic discount | Ch 9 (coming) | L3 |
| AUC | Sorting + trapezoidal integration | Ch 9 (coming) | L2 |
| Recall@K | Heap + set intersection | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L2 |
| Calibration | Histogram binning · isotonic regression | Ch 9 (coming) | L3 |
| Perplexity | Log-probability DP | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L2 |

---

## Reinforcement Learning and Planning

| ML Concept | Algorithm Inside | Chapter | Depth |
|-----------|-----------------|---------|-------|
| Value iteration | DP with Bellman equations | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L4 |
| Q-learning | DP value table · greedy policy | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L4 |
| MCTS (Monte Carlo Tree Search) | Tree search · rollouts · backprop | Ch 6 (coming) | L4 |
| Policy gradient | Sequence DP · reward accumulation | Ch 6 (coming) | L4 |
| Reward shaping | Graph-based state transitions | Ch 6 (coming) | L3 |

---

## Agents and Tool Use

| ML / Agent Concept | Algorithm Inside | Chapter | Depth |
|-------------------|-----------------|---------|-------|
| Tool registry | HashMap dispatch | [Ch 1]({% link docs/part1/chapter01-hashing.md %}) | L2 |
| Fuzzy tool name matching | Edit distance · HashMap | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L2 |
| Agent memory (short-term) | LRU Cache · HashMap | [Ch 1]({% link docs/part1/chapter01-hashing.md %}) | L2 |
| Agent memory (long-term) | Vector search · BM25 | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L3 |
| ReAct / tool-call loop | Graph BFS · state machine | Ch 5 (coming) | L3 |
| Tree of Thoughts | Beam search DP · scoring | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L4 |
| Agent DAG (planner) | Topological sort | Ch 5 (coming) | L3 |
| Multi-agent coordination | Graph message passing | Ch 5 (coming) | L4 |
| Prompt caching | HashMap · content-addressable cache | [Ch 1]({% link docs/part1/chapter01-hashing.md %}) | L3 |

---

## ML Systems and Infrastructure

| System | Algorithm / DS Inside | Chapter | Depth |
|--------|----------------------|---------|-------|
| Feature store (online) | HashMap · consistent hashing · LRU | [Ch 1]({% link docs/part1/chapter01-hashing.md %}) | L3 |
| Feature store (offline) | Distributed sort-merge join | Ch 11 (coming) | L4 |
| Model serving cache | LRU Cache · TTL HashMap | [Ch 1]({% link docs/part1/chapter01-hashing.md %}) | L3 |
| Candidate generation | ANN index · BFS on user-item graph | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L3 |
| Ranking pipeline | Heap sort · learning-to-rank | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L3 |
| A/B experiment platform | Hashing · greedy allocation | Ch 7 (coming) | L3 |
| Data deduplication | Bloom filter · MinHash · LSH | Ch 10 (coming) | L4 |
| Streaming feature monitoring | Count-Min Sketch · sliding window | Ch 10 (coming) | L4 |
| Fraud detection (graph) | Connected components · BFS | Ch 5 (coming) | L3 |

---

[← Home](/) · [DSA → ML Index →]({% link docs/index-dsa-to-ml.md %})
