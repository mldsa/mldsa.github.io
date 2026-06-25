---
layout: home
title: Home
nav_order: 1
description: "A dual-view interview guide connecting algorithms, ML, LLMs, agents, and systems."
permalink: /
---

# From LeetCode to Learning Systems
{: .fs-9 }

A dual-view guide for engineers learning ML and ML practitioners mastering algorithms.
{: .fs-5 .fw-300 }

[DSA → ML Path]({% link docs/index-dsa-to-ml.md %}){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[ML → DSA Path]({% link docs/index-ml-to-dsa.md %}){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 .mr-2 }
[GitHub](https://github.com/mldsa/learning-systems-from-scratch){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Choose your entry point

Most interview prep is asymmetric. This book fixes that — from both directions.

---

### 🔵 DSA → ML  ·  For engineers moving into ML

You know algorithms. This path shows you the ML system hiding inside every pattern.

| DSA Pattern | → | ML / LLM / Agent System |
|-------------|---|------------------------|
| HashMap | → | Tokenizer · KV Cache · Feature Store · Tool Registry |
| Heap / Top-K | → | Vector Search · RAG Retrieval · Beam Search · KNN |
| Dynamic Programming | → | Beam Search · Viterbi · RL Value Iteration · BLEU/ROUGE |
| Graph BFS / DFS | → | Recommendation · Fraud Detection · Agent DAGs |
| Trees / Tries | → | Decision Trees · XGBoost · Tokenizer Tries |
| Binary Search | → | Threshold Tuning · ANN Search · Hyperparameter Sweep |
| Recursion / Backtracking | → | Agent Planning · Tree of Thoughts · MCTS |
| Greedy / Intervals | → | Batch Scheduling · Inference Queues · Traffic Allocation |
| Streaming / Sketches | → | Feature Monitoring · Dedup · Online Metrics |
| Bit Manipulation | → | Quantization · Bloom Filters · Compressed Embeddings |

[Start: DSA → ML Index →]({% link docs/index-dsa-to-ml.md %}){: .btn .btn-primary .mr-2 }

---

### 🟢 ML → DSA  ·  For ML practitioners fixing their algorithms

You know models. This path shows you the algorithm powering every ML system.

| ML / LLM Concept | → | Algorithm That Powers It |
|------------------|---|--------------------------|
| Tokenizer (BPE) | → | HashMap · Trie · DP (Word Break) |
| Vector Search / RAG | → | Heap · Binary Search · KNN · HNSW |
| `model.generate()` | → | DP · Beam Search · Greedy Decoding |
| XGBoost / Decision Tree | → | Recursion · Greedy Split · Tree Traversal |
| NER / POS Tagging | → | Viterbi · DP · Graph CRF |
| Agent Tool Dispatch | → | HashMap · Edit Distance · Graph Routing |
| Feature Store | → | HashMap · Consistent Hashing · LRU Cache |
| Recommendation Retrieval | → | Heap · Graph BFS · ANN Index |
| A/B Testing Platform | → | Greedy · Sampling · Statistical DP |
| LLM Evaluation (ROUGE) | → | LCS · Edit Distance · DP |

[Start: ML → DSA Index →]({% link docs/index-ml-to-dsa.md %}){: .btn .btn-primary .mr-2 }

---

## Depth levels

Each topic is tagged with a suggested depth for each reader type.

| Level | Name | What it means |
|-------|------|---------------|
| **L1** | Interview Literacy | Intuition, vocabulary, common questions |
| **L2** | Implementation Ready | Can implement the core version from scratch |
| **L3** | Deep Reasoning | Math, tradeoffs, debugging |
| **L4** | Advanced / Production | Scale, optimize, extend to real systems |

[Algo Depth Track →]({% link docs/track-algo-depth.md %})  ·  [ML Depth Track →]({% link docs/track-ml-depth.md %})

---

## Chapters

### Part 0 — Orientation
- [The Asymmetric Candidate]({% link docs/part0/asymmetric-candidate.md %})

### Part 1 — Algorithm Spine

| Chapter | Topic | Notebook |
|---------|-------|---------|
| Ch 1 | [Hashing, Token Counting, and LLM Context]({% link docs/part1/chapter01-hashing.md %}) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mldsa/learning-systems-from-scratch/blob/main/notebooks/chapter01_hashing.ipynb) |
| Ch 2 | [Top-K, Vector Search, and RAG Retrieval]({% link docs/part1/chapter02-topk.md %}) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mldsa/learning-systems-from-scratch/blob/main/notebooks/chapter02_topk.ipynb) |
| Ch 3 | [Dynamic Programming — 1D, 2D, Beam Search, Viterbi, RL]({% link docs/part1/chapter03-dp.md %}) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mldsa/learning-systems-from-scratch/blob/main/notebooks/chapter03_dp.ipynb) |
| Ch 4 | Trees, Decision Trees, and XGBoost | *Coming soon* |
| Ch 5 | Graphs, Recommendation, and Fraud | *Coming soon* |

### Parts 2–8 — Coming as #MLDSA series progresses
{: .text-grey-dk-000 }

Advanced Algorithms · Statistics · Classical ML · Deep Learning · LLMs · Agents · ML Systems

---

## Author

**Deb Paul** — MLE, Seattle · Writing this publicly as the [#MLDSA](https://linkedin.com/in/dpaul0501) LinkedIn series.  
[github.com/dpaul0501](https://github.com/dpaul0501) · [LinkedIn](https://linkedin.com/in/dpaul0501)

{: .note }
Star the [GitHub repo](https://github.com/mldsa/learning-systems-from-scratch) to follow along. New chapters ship every 5 days.
