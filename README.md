# From LeetCode to Learning Systems

> A dual-view interview guide for engineers, ML engineers, applied scientists, and data scientists.

**Book site:** https://dpaul0501.github.io/learning-systems-from-scratch  
**Author:** [Deb Paul](https://github.com/dpaul0501) · [LinkedIn #MLDSA](https://linkedin.com/in/dpaul0501)

---

## What is this?

Most interview preparation is asymmetric. Engineers over-index on LeetCode. ML practitioners over-index on modeling. This book bridges both worlds.

**Every algorithm has an ML use case.**  
**Every ML model has an algorithmic implementation.**  
**Every interview topic has a system design consequence.**

## Structure

| Part | Chapters | Status |
|------|----------|--------|
| 0 · Orientation | The Asymmetric Candidate, Computational Thinking | ✅ |
| 1 · Algorithm Spine | Hashing, Top-K, DP, Trees, Graphs, Search | ✅ Ch 1–3 |
| 2 · Advanced Algorithms | Streaming, Parallel, Bits, Randomized, Optimization | 🔜 |
| 3 · Math & Statistics | Probability, Statistics, Linear Algebra, Causality | 🔜 |
| 4 · Classical ML | Regression, Trees, Boosting, Clustering, Evaluation | 🔜 |
| 5 · Deep Learning | Autograd, CNNs, Attention, Transformers | 🔜 |
| 6 · LLMs | Tokenization, Decoding, RAG, RLHF, Evaluation | 🔜 |
| 7 · Agents | Loops, Tool Use, Memory, Planning, Safety | 🔜 |
| 8 · ML Systems | Feature Stores, Serving, Recommendation, Search | 🔜 |

## How to use this

**Each chapter has two surfaces:**

- 📖 **Book page** — readable prose + code snippets on the website
- 📓 **Jupyter notebook** — runnable in Google Colab, one click

[![Open Ch1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dpaul0501/learning-systems-from-scratch/blob/main/notebooks/chapter01_hashing.ipynb)
[![Open Ch2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dpaul0501/learning-systems-from-scratch/blob/main/notebooks/chapter02_topk.ipynb)
[![Open Ch3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dpaul0501/learning-systems-from-scratch/blob/main/notebooks/chapter03_dp.ipynb)

## Run locally

```bash
git clone https://github.com/dpaul0501/learning-systems-from-scratch
cd learning-systems-from-scratch
bundle install
bundle exec jekyll serve
# → http://localhost:4000/learning-systems-from-scratch/
```

**Requirements:** Ruby 3.2+, Bundler

## Contributing

Found a bug, a better explanation, or a missing ML connection? PRs welcome.

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Following along

This book is being written publicly as the [#MLDSA](https://linkedin.com/in/dpaul0501) LinkedIn series — one chapter per 5 days, 60 days total.

---

© 2026 Deb Paul · MIT License
