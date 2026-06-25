---
layout: default
title: "ML Spine"
nav_order: 6
has_children: true
description: "ML-first deep dives: RAG, LLM internals, and Agents — for practitioners who want algorithmic depth."
---

# ML Spine
{: .no_toc }

*Deep ML coverage for practitioners — with algorithmic grounding at every step.*
{: .fs-5 .text-grey-dk-000 }

---

The Algorithm Spine (Part 1) teaches DSA with ML as the payoff. The ML Spine goes the other direction: start from the ML system, build it from scratch, and surface the algorithm powering each component.

## Chapters

| Chapter | Topic | DSA Inside |
|---------|-------|-----------|
| [ML-A]({% link docs/ml-spine/chapter-rag.md %}) | RAG in Depth — chunking, embedding, indexing, retrieval, reranking, evaluation | Heap · HashMap · BM25 |
| [ML-B]({% link docs/ml-spine/chapter-llm.md %}) | LLM Internals — BPE, attention, KV cache, decoding, LoRA, RLHF/DPO | HashMap · DP · Heap · LRU |
| [ML-C]({% link docs/ml-spine/chapter-agents.md %}) | Agents in Depth — ReAct, tools, memory, planning, evaluation | HashMap · Graph · DP · BFS |

## Who this is for

**ML practitioners** who can train a model but want to understand what's inside the black boxes they use daily — what happens inside `retriever.get_relevant_documents()`, what `model.generate()` actually does token by token, and why agent loops fail in production.

**Engineers** preparing for ML system design interviews who need ML depth to complement their DSA strength.

## How it connects to the Algorithm Spine

Every chapter in the ML Spine links back to the algorithm that powers it:

- RAG retrieval → [Ch 2: Heap / Top-K / Vector Search]({% link docs/part1/chapter02-topk.md %})
- BPE tokenization → [Ch 3: Word Break DP]({% link docs/part1/chapter03-dp.md %})
- Beam search decoding → [Ch 3: Beam Search DP]({% link docs/part1/chapter03-dp.md %})
- Agent tool registry → [Ch 1: HashMap]({% link docs/part1/chapter01-hashing.md %})
- Agent planning → [Ch 3: DP / Graph BFS]({% link docs/part1/chapter03-dp.md %})

[← Back to Home](/)
