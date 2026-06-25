---
layout: default
title: "DSA → ML Index"
nav_order: 2
description: "Start from algorithms. Find the ML system hiding inside every DSA pattern."
---

# DSA → ML Index
{: .no_toc }

*For engineers: every algorithm you know is powering an ML system you'll be asked to build.*
{: .fs-5 .text-grey-dk-000 }

---

Use this index if you think in algorithms first. Each row maps a DSA pattern you already know to the ML system it powers — and the chapter that connects them.

---

## Arrays, Strings, and Matrices

| DSA Concept | ML / LLM Connection | Chapter | Depth |
|-------------|---------------------|---------|-------|
| Sliding window | Time-series feature extraction · Context windows in LLMs | Ch 2 (coming) | L2 → L3 |
| Prefix sums | Cumulative metrics · Attention score aggregation | Ch 2 (coming) | L2 |
| Two pointers | Sequence alignment · Edit distance optimization | Ch 2 (coming) | L2 |
| Matrix traversal | Image tensors · Attention matrices · Confusion matrix | Ch 2 (coming) | L2 |
| Sparse arrays | Sparse feature vectors · Sparse embeddings | Ch 2 (coming) | L3 |

---

## Hashing and Counting

| DSA Concept | ML / LLM Connection | Chapter | Depth |
|-------------|---------------------|---------|-------|
| HashMap | Tokenizer vocabulary · Feature store lookup | [Ch 1]({% link docs/part1/chapter01-hashing.md %}) | L3 |
| LRU Cache | KV cache in LLMs · Model serving cache | [Ch 1]({% link docs/part1/chapter01-hashing.md %}) | L3 |
| Frequency map | Token counting · Bag-of-words · TF-IDF | [Ch 1]({% link docs/part1/chapter01-hashing.md %}) | L2 |
| Feature hashing | Hashing trick for high-cardinality categoricals | [Ch 1]({% link docs/part1/chapter01-hashing.md %}) | L3 |
| Bloom filter | Training data deduplication · URL dedup in crawlers | Ch 10 (coming) | L4 |
| Count-Min Sketch | Streaming token frequency · Heavy hitters in telemetry | Ch 10 (coming) | L4 |
| HyperLogLog | Unique user counting · Unique token estimation | Ch 10 (coming) | L4 |
| Consistent hashing | Distributed feature stores · Sharded embedding tables | Ch 11 (coming) | L4 |

---

## Heap and Top-K

| DSA Concept | ML / LLM Connection | Chapter | Depth |
|-------------|---------------------|---------|-------|
| Min-heap top-K | KNN retrieval · Candidate generation | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L3 |
| Max-heap | Top-K frequent tokens · Feature importance ranking | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L3 |
| Quickselect | Fast percentile computation for model monitoring | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L3 |
| Streaming top-K | Real-time leaderboard · Online top-K error tracking | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L3 |
| Merge K sorted lists | Distributed retrieval merge · Multi-index RAG | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L4 |
| Two-heap median | Online latency monitoring · Streaming metric dashboards | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L4 |
| Brute-force KNN | Vector search baseline · Embedding similarity | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L2 |
| Beam search | LLM token generation · Translation decoding | [Ch 2]({% link docs/part1/chapter02-topk.md %}) | L4 |

---

## Dynamic Programming

| DSA Concept | ML / LLM Connection | Chapter | Depth |
|-------------|---------------------|---------|-------|
| 1D DP (counting paths) | Decode Ways → tokenization segmentation | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L2 |
| 1D DP (min cost) | Coin Change → RL value iteration · Bellman equations | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L3 |
| Word Break DP | BPE tokenization · Structured output parsing | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L3 |
| Sequence DP (LIS) | Longest consistent chain in agent reasoning | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L2 |
| Edit distance (2D DP) | Spelling correction · ROUGE · WER · fuzzy tool matching | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L3 |
| LCS (2D DP) | ROUGE-L · Code similarity · Output deduplication | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L3 |
| Beam search as DP | `model.generate()` internals · Translation · Constrained decoding | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L4 |
| Viterbi algorithm | NER · POS tagging · CRF decoding · Structured LLM output | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L4 |
| Value iteration | RL planning · Agent reward optimization | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L4 |
| Tree of Thoughts DP | Agent reasoning chain scoring · Best-of-N selection | [Ch 3]({% link docs/part1/chapter03-dp.md %}) | L4 |

---

## Trees and Tries

| DSA Concept | ML / LLM Connection | Chapter | Depth |
|-------------|---------------------|---------|-------|
| Binary tree traversal | Decision tree inference · Feature selection paths | Ch 4 (coming) | L2 |
| BST | Sorted feature lookup · Threshold search | Ch 4 (coming) | L2 |
| Trie | Tokenizer prefix lookup · Autocomplete · Constrained generation | Ch 4 (coming) | L3 |
| Segment tree | Range queries in feature monitoring | Ch 4 (coming) | L3 |
| Decision tree splits | Gini / entropy · Information gain · XGBoost | Ch 4 (coming) | L4 |
| Tree recursion | Random forest training · Gradient boosting | Ch 4 (coming) | L4 |
| Histogram splits | XGBoost histogram algorithm · Approximate greedy splits | Ch 4 (coming) | L4 |

---

## Graphs

| DSA Concept | ML / LLM Connection | Chapter | Depth |
|-------------|---------------------|---------|-------|
| BFS | User-item recommendation traversal · Multi-hop retrieval | Ch 5 (coming) | L2 |
| DFS | ML pipeline dependency resolution · Agent execution trace | Ch 5 (coming) | L2 |
| Topological sort | ML pipeline DAG scheduling · Agent tool ordering | Ch 5 (coming) | L3 |
| Union-Find | Entity resolution · Duplicate user merging | Ch 5 (coming) | L3 |
| PageRank | Graph-based recommendation · Citation ranking | Ch 5 (coming) | L4 |
| Connected components | Fraud ring detection · Cluster membership | Ch 5 (coming) | L3 |
| Shortest path | Knowledge graph traversal · Multi-hop Q&A | Ch 5 (coming) | L3 |
| Random walks | Node2Vec · Graph embeddings · Recommendation | Ch 5 (coming) | L4 |

---

## Recursion and Backtracking

| DSA Concept | ML / LLM Connection | Chapter | Depth |
|-------------|---------------------|---------|-------|
| Permutations | Tool call ordering in agents · Prompt permutation | Ch 6 (coming) | L2 |
| Subsets / combinations | Feature subset selection · Ensemble construction | Ch 6 (coming) | L2 |
| Constraint search | Structured output generation · JSON schema enforcement | Ch 6 (coming) | L3 |
| Tree search (DFS) | Agent planning tree · Tree of Thoughts | Ch 6 (coming) | L4 |
| MCTS intuition | AlphaProof · o1-style reasoning · Agent planning | Ch 6 (coming) | L4 |

---

## Greedy and Intervals

| DSA Concept | ML / LLM Connection | Chapter | Depth |
|-------------|---------------------|---------|-------|
| Activity selection | Experiment traffic allocation · A/B test scheduling | Ch 7 (coming) | L2 |
| Interval merging | Chunk deduplication in RAG · Sliding window features | Ch 7 (coming) | L2 |
| Priority queue greedy | Inference queue scheduling · Batch construction | Ch 7 (coming) | L3 |
| Sweep line | Online metric monitoring · Serving SLA management | Ch 7 (coming) | L3 |

---

## Sorting and Searching

| DSA Concept | ML / LLM Connection | Chapter | Depth |
|-------------|---------------------|---------|-------|
| Binary search | Threshold tuning · Hyperparameter bisection · ANN index | Ch 2 (coming) | L2 |
| Binary search on answer | Optimal batch size search · Learning rate scheduling | Ch 2 (coming) | L3 |
| Quickselect | Online percentile for latency SLOs | Ch 2 (coming) | L3 |

---

## Bits, Math, and Probability

| DSA Concept | ML / LLM Connection | Chapter | Depth |
|-------------|---------------------|---------|-------|
| XOR / bit tricks | Bloom filter internals · Feature deduplication | Ch 12 (coming) | L3 |
| Bitsets | Bitmap indexes · Inverted index compression | Ch 12 (coming) | L3 |
| Quantization (int8) | Model quantization · Memory-efficient inference | Ch 12 (coming) | L4 |
| Reservoir sampling | Training data sampling · Streaming negative sampling | Ch 13 (coming) | L3 |
| Weighted random sampling | Exploration in bandits · Negative sampling in embeddings | Ch 13 (coming) | L3 |
| Monte Carlo | A/B test simulation · RL environment rollouts | Ch 13 (coming) | L3 |

---

[← Home](/) · [ML → DSA Index →]({% link docs/index-ml-to-dsa.md %})
