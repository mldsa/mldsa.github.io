---
layout: default
title: "ML Depth Track"
nav_order: 5
description: "Ordered by ML depth — from foundations to production systems."
---

# ML Depth Track
{: .no_toc }

*For ML practitioners: start here if you want ML depth, with the DSA implementation as the grounding at each step.*
{: .fs-5 .text-grey-dk-000 }

Read top to bottom. Each stage assumes the previous. DSA implementations are in every chapter.

---

## Stage 1 — Math and Statistical Foundations (L2–L3)
*The substrate. Without this, ML models are black boxes.*

| Topic | Key Concepts | DSA Grounding | Status |
|-------|-------------|---------------|--------|
| Probability | Distributions · Expectation · Bayes rule · CLT | Reservoir sampling · Weighted random | Ch 16 (coming) |
| Statistics | Confidence intervals · Hypothesis testing · Power | Bootstrap · Permutation test | Ch 17 (coming) |
| Linear Algebra | Vectors · Matrices · SVD · Projections | Sparse dot product · Matrix multiply | Ch 18 (coming) |
| Optimization | Gradient descent · Convexity · Adam | Iterative DP · Convergence analysis | Ch 14 (coming) |

---

## Stage 2 — Classical ML From Scratch (L3–L4)
*Every model built from scratch. No sklearn.*

| Topic | Implement From Scratch | DSA Inside | Status |
|-------|-----------------------|-----------|--------|
| Linear Regression | Gradient descent · Ridge · closed form | Matrix multiply · iterative update | Ch 21 (coming) |
| Logistic Regression | SGD · cross-entropy · calibration | Sigmoid · log-space arithmetic | Ch 22 (coming) |
| KNN | Euclidean distance · top-K | **Heap top-K** · brute-force search | [Ch 2 →]({% link docs/part1/chapter02-topk.md %}) |
| Decision Tree | Gini · entropy · greedy splits | Tree recursion · threshold scan | Ch 25 (coming) |
| Random Forest | Bagging · feature subsampling · OOB error | Parallel tree recursion · sampling | Ch 26 (coming) |
| Gradient Boosting | Residual learning · additive models | Greedy · iterative DP | Ch 27 (coming) |
| XGBoost | Taylor expansion · histogram splits | Histogram algorithm · split gain | Ch 27 (coming) |
| K-Means | Lloyd's algorithm · centroid update | Greedy assignment · heap distance | Ch 28 (coming) |
| PCA | Covariance · eigenvectors · SVD | Matrix ops · dimensionality reduction | Ch 28 (coming) |

---

## Stage 3 — Model Evaluation (L2–L3)
*Before deep learning: learn how to measure models correctly.*

| Topic | Key Concepts | DSA / Algo Inside | Status |
|-------|-------------|-------------------|--------|
| Classification metrics | Precision · Recall · F1 · AUC · ROC | Sorting · threshold scan | Ch 29 (coming) |
| Ranking metrics | NDCG · MAP · MRR · Recall@K | **Heap top-K** · sorted list | [Ch 2 →]({% link docs/part1/chapter02-topk.md %}) |
| Regression metrics | MSE · MAE · R² | Array ops | Ch 29 (coming) |
| Sequence metrics | ROUGE-L · WER · BLEU | **Edit distance · LCS** | [Ch 3 →]({% link docs/part1/chapter03-dp.md %}) |
| Calibration | Reliability diagram · isotonic regression | Histogram · sorting | Ch 29 (coming) |

---

## Stage 4 — Deep Learning From Scratch (L3–L4)
*Build autograd, then build everything on top of it.*

| Topic | Implement From Scratch | DSA Inside | Status |
|-------|-----------------------|-----------|--------|
| Neural Network Foundations | MLP · activations · forward pass | Matrix multiply · batching | Ch 30 (coming) |
| **Autograd Engine** | Scalar → tensor · chain rule · topological sort | **DAG topological sort** | Ch 31 (coming) |
| Backpropagation | Chain rule · gradient accumulation | Reverse graph traversal | Ch 31 (coming) |
| Softmax + Cross-Entropy | Stable softmax · log-sum-exp | Numerical stability tricks | Ch 32 (coming) |
| CNN | 2D convolution · pooling · batch norm | **Sliding window** · matrix ops | Ch 33 (coming) |
| RNN / LSTM | Sequential hidden state · gating | **Sequence DP** · chain rule | Ch 34 (coming) |
| Attention | QKV · scaled dot-product · causal mask | **Matrix multiply · Heap** | Ch 35 (coming) |
| Transformer | Multi-head attention · positional encoding | Full graph computation | Ch 35 (coming) |

---

## Stage 5 — LLMs (L3–L4)
*From architecture to alignment.*

| Topic | Key Concepts | DSA Inside | Status |
|-------|-------------|-----------|--------|
| Tokenization | BPE · WordPiece · vocabulary | **HashMap · Trie · DP** | [Ch 1 →]({% link docs/part1/chapter01-hashing.md %}), [Ch 3 →]({% link docs/part1/chapter03-dp.md %}) |
| Transformer LLM | Decoder-only · KV cache · parameter count | **LRU Cache · Matrix ops** | [Ch 1 →]({% link docs/part1/chapter01-hashing.md %}), Ch 38 (coming) |
| Decoding algorithms | Greedy · Beam · Top-k · Top-p | **DP · Heap** | [Ch 3 →]({% link docs/part1/chapter03-dp.md %}), [Ch 2 →]({% link docs/part1/chapter02-topk.md %}) |
| Fine-tuning · LoRA | Full FT · adapters · catastrophic forgetting | Matrix ops | Ch 41 (coming) |
| RLHF · DPO | Preference data · reward models · alignment | DP reward accumulation | Ch 42 (coming) |
| RAG | Chunking · retrieval · reranking · grounding | **Heap · ANN · HashMap** | [Ch 2 →]({% link docs/part1/chapter02-topk.md %}), Ch 43 (coming) |
| LLM Evaluation | Factuality · hallucination · LLM-as-judge | **Edit distance · LCS** | [Ch 3 →]({% link docs/part1/chapter03-dp.md %}), Ch 44 (coming) |

---

## Stage 6 — Agents (L3–L4)
*LLMs with tools, memory, and planning.*

| Topic | Key Concepts | DSA Inside | Status |
|-------|-------------|-----------|--------|
| Agent loop | ReAct · observe-think-act | Graph state machine | Ch 46 (coming) |
| Tool use | Function calling · JSON schema · retries | **HashMap dispatch** | [Ch 1 →]({% link docs/part1/chapter01-hashing.md %}), Ch 47 (coming) |
| Agent memory | Short-term · long-term · vector memory | **LRU · HashMap · Vector search** | [Ch 1 →]({% link docs/part1/chapter01-hashing.md %}), Ch 48 (coming) |
| Planning | Tree of Thoughts · beam search · MCTS | **DP · Beam search** | [Ch 3 →]({% link docs/part1/chapter03-dp.md %}), Ch 49 (coming) |
| Agent evaluation | Trace eval · tool correctness · safety | Scoring DP | Ch 51 (coming) |

---

## Stage 7 — ML Systems (L3–L4)
*Build the infrastructure models live in.*

| System | What to design | DSA Inside | Status |
|--------|---------------|-----------|--------|
| Feature store | Online/offline · freshness · leakage | **HashMap · consistent hashing** | [Ch 1 →]({% link docs/part1/chapter01-hashing.md %}), Ch 53 (coming) |
| Model serving | Batching · caching · canary | **LRU · queue** | Ch 54 (coming) |
| Recommendation | Candidate gen · ranking · feedback loops | **Heap · graph · ANN** | [Ch 2 →]({% link docs/part1/chapter02-topk.md %}), Ch 55 (coming) |
| Search | Inverted index · BM25 · hybrid | **HashMap · heap** | Ch 56 (coming) |
| Anomaly detection | Z-score · Isolation Forest · drift | **Streaming sketches** | Ch 57 (coming) |
| Experimentation | A/B · power · SRM · causal inference | **Hashing · sampling** | Ch 58 (coming) |

---

## Recommended schedule (ML-inclined)

| Week | Focus | Target |
|------|-------|--------|
| 1 | DSA catch-up: Ch 1 + Ch 2 | Solve 15 LeetCode problems, implement KNN and LRU |
| 2 | DSA catch-up: Ch 3 | DP patterns, implement edit distance + beam search |
| 3–4 | Classical ML from scratch | Linear reg, logistic reg, decision tree, XGBoost |
| 5–6 | Autograd + deep learning | Autograd engine, MLP, transformer block |
| 7–8 | LLMs + RAG | Tokenizer, decoding, RAG from scratch |
| 9–10 | Agents | Agent loop, tool use, trace evaluation |
| 11–12 | ML systems design | Feature store, serving, recommendation, search |

---

[← Home](/) · [Algo Depth Track →]({% link docs/track-algo-depth.md %})
