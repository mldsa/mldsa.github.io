# From LeetCode to Learning Systems

> A dual-view interview guide for engineers, ML engineers, applied scientists, and data scientists.

**Book site:** https://mldsa.github.io  
**Author:** [Deb Paul](https://github.com/dpaul0501) · [LinkedIn #MLDSA](https://linkedin.com/in/dpaul0501)

---

## What is this?

Most interview preparation is asymmetric. Engineers over-index on LeetCode. ML practitioners over-index on modeling. This book bridges both worlds.

**Every algorithm has an ML use case.**  
**Every ML model has an algorithmic implementation.**  
**Every interview topic has a system design consequence.**

---

## Structure

### Algorithm Spine — DSA-first, ML as the payoff

| Part | Chapters | Status |
|------|----------|--------|
| 0 · Orientation | The Asymmetric Candidate | ✅ |
| 1 · Algorithm Spine | Hashing · Top-K · Dynamic Programming | ✅ Ch 1–3 |
| 2 · Advanced Algorithms | Streaming · Parallel · Bits · Randomized · Optimization | 🔜 |

### ML Spine — ML-first, algorithm as the foundation

| Chapter | Topic | Status |
|---------|-------|--------|
| ML-A | [RAG in Depth](https://mldsa.github.io/docs/ml-spine/chapter-rag/) — chunking · embedding · indexing · reranking · evaluation | ✅ |
| ML-B | [LLM Internals](https://mldsa.github.io/docs/ml-spine/chapter-llm/) — BPE · attention · KV cache · decoding · LoRA · DPO | ✅ |
| ML-C | [Agents in Depth](https://mldsa.github.io/docs/ml-spine/chapter-agents/) — ReAct · tools · memory · planning · evaluation | ✅ |
| 3 · Math & Statistics | Probability · Statistics · Linear Algebra · Causality | 🔜 |
| 4 · Classical ML | Regression · Trees · Boosting · Clustering · Evaluation | 🔜 |
| 5 · Deep Learning | Autograd · CNNs · Attention · Transformers | 🔜 |
| 6 · LLMs | Tokenization · Decoding · RAG · RLHF · Evaluation | 🔜 |
| 7 · Agents | Loops · Tool Use · Memory · Planning · Safety | 🔜 |
| 8 · ML Systems | Feature Stores · Serving · Recommendation · Search | 🔜 |

### Dual index

| Index | Description |
|-------|-------------|
| [DSA → ML](https://mldsa.github.io/docs/index-dsa-to-ml/) | Every algorithm mapped to the ML system it powers |
| [ML → DSA](https://mldsa.github.io/docs/index-ml-to-dsa/) | Every ML concept mapped to the algorithm inside it |
| [Algo Depth Track](https://mldsa.github.io/docs/track-algo-depth/) | 4-stage reading path for engineers |
| [ML Depth Track](https://mldsa.github.io/docs/track-ml-depth/) | 7-stage reading path for ML practitioners |

---

## How to use this

Each chapter has two surfaces:

- 📖 **Book page** — prose + code snippets on the website
- 📓 **Jupyter notebook** — runnable in Google Colab, one click

[![Open Ch1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mldsa/mldsa.github.io/blob/main/notebooks/chapter01_hashing.ipynb)
[![Open Ch2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mldsa/mldsa.github.io/blob/main/notebooks/chapter02_topk.ipynb)
[![Open Ch3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mldsa/mldsa.github.io/blob/main/notebooks/chapter03_dp.ipynb)

---

## Run locally

```bash
git clone https://github.com/mldsa/mldsa.github.io
cd mldsa.github.io
bundle install
bundle exec jekyll serve
# → http://localhost:4000/
```

**Requirements:** Ruby 3.2+, Bundler

---

## Contributing

Found a bug, a better explanation, or a missing ML connection? PRs welcome.

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Following along

This book is being written publicly as the [#MLDSA](https://linkedin.com/in/dpaul0501) LinkedIn series — one chapter every 5 days, 60 days total.

⭐ Star this repo to follow along.

---

© 2026 Deb Paul · MIT License
