---
layout: default
title: "Algo Depth Track"
nav_order: 4
description: "Ordered by algorithmic depth — from easy patterns to advanced systems."
---

# Algo Depth Track
{: .no_toc }

*For engineers: start here if you want full algorithmic depth, with ML as the payoff at the end of each topic.*
{: .fs-5 .text-grey-dk-000 }

Read top to bottom. Each row builds on the previous. ML connections are in every chapter.

---

## Stage 1 — Foundation Patterns (L1–L2)
*Cover these first. They appear in 80% of MLE coding rounds.*

| Topic | Core Problems | ML Payoff | Status |
|-------|--------------|-----------|--------|
| HashMap · Counting · Grouping | Two Sum, Group Anagrams, LRU Cache | Tokenizers · Feature stores · KV cache | [Ch 1 →]({% link docs/part1/chapter01-hashing.md %}) |
| Heap · Top-K · Streaming | Kth Largest, Top K Frequent, Median Stream | Vector search · RAG retrieval · Beam search | [Ch 2 →]({% link docs/part1/chapter02-topk.md %}) |
| Arrays · Sliding Window · Prefix Sum | Max Subarray, Sliding Window Max | Feature windows · Convolution intuition | Ch 2 (coming) |
| Binary Search · Search on Answer | Binary Search, Search Rotated Array | Threshold tuning · Hyperparameter search | Ch 2 (coming) |
| Stack · Queue · Monotonic | Valid Parentheses, Largest Rectangle | Parser state · Constrained generation | Ch 2 (coming) |

---

## Stage 2 — Core Patterns (L2–L3)
*Unlock the hard mediums. These appear in every ML system design.*

| Topic | Core Problems | ML Payoff | Status |
|-------|--------------|-----------|--------|
| Trees · BFS · DFS | Tree traversal, Level Order, Lowest Common Ancestor | Decision trees · XGBoost · Trie tokenizer | Ch 4 (coming) |
| Tries | Implement Trie, Word Search II | Autocomplete · Constrained LLM decoding | Ch 4 (coming) |
| Graphs · Connected Components | Islands, Clone Graph, Accounts Merge | Recommendation · Fraud detection | Ch 5 (coming) |
| Union-Find | Accounts Merge, Redundant Connection | Entity resolution · Duplicate merging | Ch 5 (coming) |
| 1D Dynamic Programming | Climbing Stairs, Coin Change, Word Break | RL value iteration · Tokenization | [Ch 3 →]({% link docs/part1/chapter03-dp.md %}) |
| Greedy · Intervals | Merge Intervals, Meeting Rooms | Experiment scheduling · Inference batching | Ch 7 (coming) |
| Topological Sort | Course Schedule, Alien Dictionary | ML pipeline DAGs · Agent planning | Ch 5 (coming) |

---

## Stage 3 — Hard Patterns (L3–L4)
*These separate mediors from seniors. Appear in system design extensions.*

| Topic | Core Problems | ML Payoff | Status |
|-------|--------------|-----------|--------|
| 2D Dynamic Programming | Edit Distance, LCS, Longest Palindrome | ROUGE-L · WER · Sequence alignment | [Ch 3 →]({% link docs/part1/chapter03-dp.md %}) |
| Recursion · Backtracking | Subsets, Permutations, N-Queens | Agent planning · Tool-call enumeration | Ch 6 (coming) |
| Shortest Path · Dijkstra | Network Delay, Cheapest Flights | Knowledge graph traversal · Multi-hop QA | Ch 5 (coming) |
| Segment Tree · Fenwick | Range Sum Query, Count of Smaller | Streaming range queries on features | Ch 4 (coming) |
| Bit Manipulation | Single Number, Count Bits, Bitset | Bloom filters · Quantization · Compression | Ch 12 (coming) |
| Randomized Algorithms | Random Pick with Weight, Shuffle | Reservoir sampling · Dropout · Bandits | Ch 13 (coming) |

---

## Stage 4 — Advanced / Production (L4)
*Rarely tested in coding rounds but essential for ML system design interviews.*

| Topic | Core Problems | ML Payoff | Status |
|-------|--------------|-----------|--------|
| Streaming · Sketches | — | Bloom filter · Count-Min · HyperLogLog | Ch 10 (coming) |
| Beam Search (DP) | — | LLM decoding · Constrained output | [Ch 3 →]({% link docs/part1/chapter03-dp.md %}) |
| Viterbi (Sequence DP) | — | NER · CRF decoding · Structured output | [Ch 3 →]({% link docs/part1/chapter03-dp.md %}) |
| ANN Index (HNSW) | — | Vector DB · RAG at scale | [Ch 2 →]({% link docs/part1/chapter02-topk.md %}) |
| Parallel / Distributed | — | Distributed training · Batch inference | Ch 11 (coming) |
| Optimization Algorithms | — | Gradient descent · Adam · LR schedules | Ch 14 (coming) |

---

## Recommended weekly schedule (engineer-inclined)

| Week | Focus | Target |
|------|-------|--------|
| 1–2 | Stage 1 foundation | Ch 1 + Ch 2, 20 LeetCode problems |
| 3–4 | Stage 2 trees + graphs | Ch 4 + Ch 5, 30 problems |
| 5–6 | Stage 2 DP + greedy | Ch 3 + Ch 7, 25 problems |
| 7–8 | Stage 3 hard patterns | Ch 3 (2D DP), 20 problems |
| 9–10 | ML from scratch | Ch 4 (Classical ML), implementations |
| 11–12 | Stage 4 + ML systems | System design + advanced chapters |

---

[← Home](/) · [ML Depth Track →]({% link docs/track-ml-depth.md %})
