---
layout: default
title: "ML-B · LLM Internals"
parent: "ML Spine"
nav_order: 2
description: "How LLMs actually work: BPE tokenization, attention, KV cache, decoding algorithms, fine-tuning, and alignment."
---

# LLM Internals
{: .no_toc }

*ML Spine · Chapter B*
{: .text-grey-dk-000 .fs-4 }

<div class="reading-meta">
  ⏱ 40 min read &nbsp;·&nbsp;
  🟩 ML: L3–L4 &nbsp;·&nbsp;
  🟦 DSA inside: HashMap · DP · Heap · LRU Cache
</div>

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. The journey of a prompt

Before writing a single line of training code, understand what happens to your input:

```
"What is the capital of France?"
        ↓
[Tokenizer]     → [1867, 374, 279, 6864, 315, 9822, 30]
        ↓
[Embedding]     → matrix (7 tokens × d_model floats)
        ↓
[Transformer]   → 32–96 layers of attention + FFN
        ↓
[LM Head]       → logit vector over vocabulary (~50k entries)
        ↓
[Decoding]      → sample or argmax → next token id
        ↓
[Detokenizer]   → "Paris"
```

This chapter works through each stage with implementations.

---

## 2. Tokenization: BPE from scratch

### Why not just use characters or words?

- **Characters:** vocabulary too small → sequences too long → attention is O(n²) → too slow
- **Words:** vocabulary too large → rare words get no training signal → can't handle typos or new words
- **Subwords (BPE):** sweet spot — common words are single tokens, rare words split into known pieces

### Byte Pair Encoding

BPE starts with individual characters and iteratively merges the most frequent adjacent pair:

```python
from collections import defaultdict, Counter

def train_bpe(corpus: list[str], num_merges: int = 100) -> tuple[dict, list[tuple]]:
    """
    Train BPE tokenizer from scratch.
    
    Returns:
      vocab: dict mapping token → id
      merges: list of (token_a, token_b) merge rules in order
    
    DSA inside: HashMap frequency counting + greedy merge (like Huffman coding)
    """
    # Step 1: Initialize vocabulary with characters + end-of-word marker
    # Represent each word as a tuple of characters
    word_freqs = Counter(corpus)
    vocab = defaultdict(int)   # token → frequency

    # Split words into character sequences
    splits = {word: list(word) + ['</w>'] for word in word_freqs}

    merges = []
    base_tokens = set()
    for word in splits:
        for ch in splits[word]:
            base_tokens.add(ch)

    for merge_num in range(num_merges):
        # Count all adjacent pairs
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            chars = splits[word]
            for i in range(len(chars) - 1):
                pair_freqs[(chars[i], chars[i+1])] += freq

        if not pair_freqs:
            break

        # Find most frequent pair
        best_pair = max(pair_freqs, key=pair_freqs.get)
        merges.append(best_pair)

        # Apply merge to all words
        new_token = best_pair[0] + best_pair[1]
        for word in splits:
            chars = splits[word]
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i+1]) == best_pair:
                    new_chars.append(new_token)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            splits[word] = new_chars

    # Build vocabulary: base tokens + all merged tokens, in order of creation
    all_tokens = list(base_tokens)
    for a, b in merges:
        all_tokens.append(a + b)
    vocab = {tok: i for i, tok in enumerate(all_tokens)}
    return vocab, merges


def tokenize_bpe(text: str, vocab: dict, merges: list[tuple]) -> list[int]:
    """
    Apply trained BPE merges to tokenize new text.
    
    DSA inside: Word Break DP — find valid segmentation using learned merges.
    """
    words = text.split()
    token_ids = []
    merge_rank = {pair: i for i, pair in enumerate(merges)}

    for word in words:
        chars = list(word) + ['</w>']
        # Iteratively apply merges in rank order (lowest rank = earliest merge = highest priority)
        while len(chars) > 1:
            pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
            best = min(pairs, key=lambda p: merge_rank.get(p, float('inf')))
            if best not in merge_rank:
                break
            new_token = best[0] + best[1]
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars)-1 and (chars[i], chars[i+1]) == best:
                    new_chars.append(new_token)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        token_ids.extend(vocab.get(ch, vocab.get('<unk>', 0)) for ch in chars)
    return token_ids
```

### Token counting

A practical skill: estimate how many tokens a string costs before calling an LLM API.

```python
def rough_token_count(text: str) -> int:
    """
    Rough estimate: ~4 characters per token for English.
    Use tiktoken for exact counts (pip install tiktoken).
    """
    return len(text) // 4

# Exact:
# import tiktoken
# enc = tiktoken.encoding_for_model("gpt-4o")
# len(enc.encode(text))
```

| Content | Tokens (approx) |
|---------|-----------------|
| 1 word | ~1.3 tokens |
| 1 sentence (15 words) | ~20 tokens |
| 1 page (500 words) | ~650 tokens |
| 1 book (100k words) | ~130k tokens |
| GPT-4o context limit | 128k tokens ≈ 200 pages |

---

## 3. Attention mechanism

### Scaled dot-product attention

Attention lets every token attend to every other token and aggregate their values weighted by similarity:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- **Q** (queries): what each token is looking for
- **K** (keys): what each token offers
- **V** (values): what each token will contribute if selected
- **√d_k**: scaling factor to prevent dot products from getting too large (causing softmax saturation)

```python
import math

def scaled_dot_product_attention(
    Q: list[list[float]],   # (seq_len, d_k)
    K: list[list[float]],   # (seq_len, d_k)
    V: list[list[float]],   # (seq_len, d_v)
    mask: list[list[bool]] = None,  # True = masked out (causal mask for decoder)
) -> list[list[float]]:
    """
    Core attention operation. O(n² × d) time and space.
    
    DSA insight: this is a weighted sum where weights are
    similarity scores between queries and keys — essentially
    a soft Top-K retrieval over the sequence.
    """
    seq_len = len(Q)
    d_k = len(Q[0])
    scale = math.sqrt(d_k)

    # Step 1: Compute attention scores QK^T / √d_k
    scores = [[0.0] * seq_len for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(seq_len):
            scores[i][j] = sum(Q[i][k] * K[j][k] for k in range(d_k)) / scale

    # Step 2: Apply causal mask (decoder: token i can only see tokens 0..i)
    if mask:
        NEG_INF = float('-inf')
        for i in range(seq_len):
            for j in range(seq_len):
                if mask[i][j]:
                    scores[i][j] = NEG_INF

    # Step 3: Softmax over keys dimension
    def softmax(row):
        max_val = max(row)
        exps = [math.exp(x - max_val) for x in row]  # numerical stability
        total = sum(exps)
        return [e / total for e in exps]

    weights = [softmax(scores[i]) for i in range(seq_len)]

    # Step 4: Weighted sum of values
    d_v = len(V[0])
    output = [[sum(weights[i][j] * V[j][k] for j in range(seq_len))
               for k in range(d_v)]
              for i in range(seq_len)]
    return output
```

### Why attention is O(n²)

For a sequence of n tokens, every token attends to every other token: n × n attention scores. This is why long context is expensive:

| Sequence length | Attention matrix size | Memory |
|----------------|----------------------|--------|
| 1k tokens | 1M entries | ~4MB (float32) |
| 8k tokens | 64M entries | ~256MB |
| 128k tokens | 16B entries | ~64GB |

Flash Attention (the standard in production) reduces memory from O(n²) to O(n) by computing attention in tiles that fit in GPU SRAM, never materializing the full matrix.

### Multi-head attention

Run H attention heads in parallel, each with different learned Q/K/V projections, then concatenate outputs:

```python
def multi_head_attention_intuition(x, num_heads, head_dim):
    """
    Intuition only — production uses batched matrix ops.
    
    Each head specializes: one might attend to syntax,
    another to coreference, another to semantics.
    """
    outputs = []
    for head in range(num_heads):
        # Each head has its own W_Q, W_K, W_V matrices (learned)
        Q = project(x, W_Q[head])    # (seq_len, head_dim)
        K = project(x, W_K[head])
        V = project(x, W_V[head])
        head_output = scaled_dot_product_attention(Q, K, V, causal_mask)
        outputs.append(head_output)
    # Concatenate and project: (seq_len, num_heads × head_dim) → (seq_len, d_model)
    return project(concatenate(outputs), W_O)
```

---

## 4. KV Cache

### The problem: autoregressive generation is slow

At inference, the LLM generates one token at a time. To generate token t, it needs attention scores between token t and all previous tokens 0..t-1. Without caching, this recomputes K and V for all previous tokens at every step.

Cost without cache: generating 1000 tokens costs O(1000²) attention operations = 500k redundant computations.

### The solution: cache K and V per layer

```python
class KVCache:
    """
    Cache key-value pairs per transformer layer.
    DSA: essentially a LRU cache indexed by (layer, position).
    
    Memory: batch_size × num_layers × seq_len × 2 × num_heads × head_dim × 4 bytes
    For LLaMA-2 70B, seq_len=4096: ~140GB peak KV cache. Why batching matters.
    """
    def __init__(self, num_layers: int, max_seq_len: int):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        # cache[layer] = {"k": [...], "v": [...]}
        self.cache = {layer: {"k": [], "v": []} for layer in range(num_layers)}
        self.current_len = 0

    def update(self, layer: int, new_k: list, new_v: list):
        """Append new key-value pair for position current_len."""
        self.cache[layer]["k"].append(new_k)
        self.cache[layer]["v"].append(new_v)

    def get(self, layer: int) -> tuple[list, list]:
        """Return all cached keys and values for this layer."""
        return self.cache[layer]["k"], self.cache[layer]["v"]

    def step(self):
        self.current_len += 1

    def evict_oldest(self):
        """Sliding window attention: evict position 0 when cache is full."""
        if self.current_len >= self.max_seq_len:
            for layer in range(self.num_layers):
                self.cache[layer]["k"].pop(0)
                self.cache[layer]["v"].pop(0)
            self.current_len -= 1
```

### KV cache sizing in production

```python
def kv_cache_memory_gb(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,   # float16 = 2 bytes
) -> float:
    """
    KV cache memory budget.
    Example: LLaMA-3 70B, batch=1, seq=8192, layers=80, heads=8, head_dim=128
      → 2 × 80 × 8192 × 8 × 128 × 2 bytes = ~2.68 GB
    """
    elements = 2 * num_layers * seq_len * num_heads * head_dim
    return (elements * batch_size * dtype_bytes) / (1024 ** 3)

# Continuous batching (vLLM): share KV cache across variable-length sequences
# PagedAttention: manage KV cache like virtual memory pages → 3-4x throughput improvement
```

---

## 5. Decoding algorithms

### Greedy decoding

At each step, pick the token with the highest probability:

```python
def greedy_decode(logits_fn, input_ids: list[int], max_new_tokens: int = 100,
                  eos_token_id: int = 2) -> list[int]:
    """
    Simplest decoding. Fast. Deterministic. Often repetitive.
    logits_fn: (input_ids) → list of logits over vocab
    """
    generated = []
    current = input_ids[:]
    for _ in range(max_new_tokens):
        logits = logits_fn(current)
        next_token = logits.index(max(logits))   # argmax
        if next_token == eos_token_id:
            break
        generated.append(next_token)
        current.append(next_token)
    return generated
```

### Temperature sampling

Divide logits by temperature T before softmax. T < 1 sharpens the distribution (more deterministic), T > 1 flattens it (more random):

```python
import math, random

def softmax_with_temp(logits: list[float], temperature: float = 1.0) -> list[float]:
    if temperature == 0:
        # Greedy: return one-hot at argmax
        max_idx = logits.index(max(logits))
        return [1.0 if i == max_idx else 0.0 for i in range(len(logits))]
    scaled = [l / temperature for l in logits]
    max_l = max(scaled)
    exps = [math.exp(l - max_l) for l in scaled]
    total = sum(exps)
    return [e / total for e in exps]

def sample_token(probs: list[float]) -> int:
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i
    return len(probs) - 1
```

### Top-K sampling

Only sample from the K most probable tokens. Eliminates low-probability "tail" tokens that cause incoherence:

```python
import heapq

def top_k_sample(logits: list[float], k: int = 50, temperature: float = 1.0) -> int:
    """
    DSA inside: Heap top-K (Ch 2) applied to token sampling.
    k=50 is a common default. Lower k = more focused/deterministic.
    """
    # Get top-K indices by logit value
    top_k_pairs = heapq.nlargest(k, enumerate(logits), key=lambda x: x[1])
    top_k_ids = [idx for idx, _ in top_k_pairs]
    top_k_logits = [logits[idx] for idx in top_k_ids]
    probs = softmax_with_temp(top_k_logits, temperature)
    chosen_pos = sample_token(probs)
    return top_k_ids[chosen_pos]
```

### Top-P (nucleus) sampling

Sample from the smallest set of tokens whose cumulative probability exceeds P. Adapts dynamically — uses fewer tokens when distribution is peaked, more when it's flat:

```python
def top_p_sample(logits: list[float], p: float = 0.9, temperature: float = 1.0) -> int:
    """
    p=0.9 means: sample from tokens that together cover 90% of probability mass.
    DSA inside: sort + prefix sum + binary search (Ch 3 DP).
    """
    probs = softmax_with_temp(logits, temperature)
    # Sort by probability descending
    sorted_pairs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    # Build cumulative sum, include until cumulative >= p
    cumsum = 0.0
    nucleus_ids, nucleus_probs = [], []
    for idx, prob in sorted_pairs:
        nucleus_ids.append(idx)
        nucleus_probs.append(prob)
        cumsum += prob
        if cumsum >= p:
            break
    # Renormalize and sample
    total = sum(nucleus_probs)
    renorm_probs = [pr / total for pr in nucleus_probs]
    chosen_pos = sample_token(renorm_probs)
    return nucleus_ids[chosen_pos]
```

### Beam search

See [Ch 3]({% link docs/part1/chapter03-dp.md %}) for the full DP implementation. Key parameters for LLM use:

| Parameter | Effect | Typical value |
|-----------|--------|---------------|
| `beam_width` | More beams → better quality, more compute | 4–10 for translation, 1 for chat |
| `length_penalty` | α > 1 favors longer sequences; α < 1 favors shorter | 0.6–1.2 |
| `no_repeat_ngram_size` | Block repeating n-grams | 3–4 |
| `early_stopping` | Stop when all beams hit EOS | True for short outputs |

### When to use which

| Task | Recommended | Why |
|------|------------|-----|
| Chat / creative writing | Top-P (p=0.9), T=0.7–1.0 | Diversity, naturalness |
| Code generation | Greedy or T=0.2 | Determinism, correctness |
| Translation | Beam search (width=4) | Quality + length control |
| Math / reasoning | Best-of-N sampling, T=0.8 | Explore then verify |
| Structured output | Constrained decoding | Force valid JSON/schema |

---

## 6. Transformer architecture in numbers

### Parameter counting

```python
def count_transformer_params(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,        # FFN hidden dimension, typically 4 × d_model
) -> dict:
    """Count parameters in a decoder-only transformer."""
    # Embedding table
    embedding = vocab_size * d_model

    per_layer = {
        # Self-attention: Q, K, V projections + output projection
        "attention": 4 * d_model * d_model,
        # Feed-forward network: two linear layers
        "ffn": 2 * d_model * d_ff,
        # Layer norms (2 per layer, each has scale + bias = 2 × d_model)
        "layer_norm": 4 * d_model,
    }
    total_per_layer = sum(per_layer.values())

    total = embedding + num_layers * total_per_layer
    return {
        "embedding": embedding,
        "per_layer": per_layer,
        "total_layer_params": num_layers * total_per_layer,
        "total": total,
        "total_B": total / 1e9,
    }

# LLaMA-3 8B (approximate):
params = count_transformer_params(
    vocab_size=128_256, d_model=4096, num_layers=32,
    num_heads=32, d_ff=14336
)
print(f"Total: {params['total_B']:.1f}B parameters")  # ≈ 8.0B
```

### Memory for inference

```python
def inference_memory_gb(
    param_count: int,
    dtype_bytes: int = 2,     # float16
    kv_cache_gb: float = 2.0, # from kv_cache_memory_gb()
    activation_multiplier: float = 1.2,  # activations during forward pass
) -> float:
    """Estimate total GPU memory needed to serve a model."""
    model_weights = (param_count * dtype_bytes) / (1024 ** 3)
    return (model_weights + kv_cache_gb) * activation_multiplier

# LLaMA-3 8B, float16, seq=4096:
mem = inference_memory_gb(8e9, dtype_bytes=2, kv_cache_gb=1.5)
print(f"GPU memory needed: {mem:.1f}GB")  # ≈ 19.2 GB → needs 1× A100 40GB
```

---

## 7. Fine-tuning

### Full fine-tuning vs. LoRA

**Full fine-tuning:** update all parameters. Expensive in memory and compute. Risk of catastrophic forgetting.

**LoRA (Low-Rank Adaptation):** freeze base model. Add small trainable matrices to each attention layer. Only train those.

```python
class LoRALayer:
    """
    LoRA adds two small matrices A and B to each weight matrix W.
    Forward: h = Wx + (B @ A)x × (alpha / rank)
    
    Only A and B are trained. W is frozen.
    
    Memory savings: instead of training d_model × d_model = 4M params per layer,
    train 2 × d_model × rank params.
    
    For d_model=4096, rank=16: 4096×4096=16.7M → 2×4096×16=131k (127× fewer)
    """
    def __init__(self, d_in: int, d_out: int, rank: int = 16, alpha: float = 16):
        self.rank = rank
        self.scale = alpha / rank
        # Initialize A with random Gaussian, B with zeros
        # (ensures LoRA delta = 0 at initialization → stable training start)
        self.A = [[random.gauss(0, 0.02) for _ in range(rank)] for _ in range(d_in)]
        self.B = [[0.0] * d_out for _ in range(rank)]

    def forward(self, x: list[float], W_output: list[float]) -> list[float]:
        """Add LoRA delta to base model output."""
        # Ax: (rank,)
        Ax = [sum(x[i] * self.A[i][r] for i in range(len(x))) for r in range(self.rank)]
        # BAx: (d_out,)
        BAx = [sum(Ax[r] * self.B[r][j] for r in range(self.rank)) for j in range(len(self.B[0]))]
        # W_output + scale × BAx
        return [W_output[j] + self.scale * BAx[j] for j in range(len(W_output))]
```

### LoRA practical guide

| Parameter | Effect | Default |
|-----------|--------|---------|
| `rank` (r) | Adapter capacity. Higher = more parameters. | 8–64 |
| `alpha` | Effective learning rate scale = alpha/rank | = rank (common) |
| `target_modules` | Which layers to adapt | q_proj, v_proj |
| `dropout` | Regularization | 0.05 |

QLoRA extends LoRA by quantizing the base model to 4-bit (NF4), enabling fine-tuning of 70B models on a single A100.

---

## 8. RLHF and DPO

### Why instruction tuning isn't enough

After supervised fine-tuning on instruction-following data, models can still produce harmful, unhelpful, or verbose outputs. The base SFT model maximizes likelihood of training demonstrations — it doesn't directly optimize for human preference.

### RLHF pipeline

```
Step 1: SFT (Supervised Fine-Tuning)
  → fine-tune on high-quality (prompt, response) pairs
  → produces SFT model

Step 2: Reward Model Training
  → collect human preferences: for each prompt, rank completions A vs. B
  → train a classifier: P(A ≻ B | prompt) = σ(r(prompt, A) - r(prompt, B))
  → produces reward model r_φ

Step 3: PPO (Proximal Policy Optimization)
  → optimize the SFT model to maximize reward while staying close to SFT
  → objective: maximize E[r_φ(x, y)] - β × KL(π_θ || π_SFT)
  → KL penalty prevents reward hacking
  → produces RLHF model
```

### DPO (Direct Preference Optimization)

DPO skips the explicit reward model. It directly optimizes the policy on preference pairs:

```python
def dpo_loss(
    log_prob_chosen_policy: float,      # log π_θ(y_w | x)
    log_prob_rejected_policy: float,    # log π_θ(y_l | x)
    log_prob_chosen_ref: float,         # log π_ref(y_w | x)
    log_prob_rejected_ref: float,       # log π_ref(y_l | x)
    beta: float = 0.1,
) -> float:
    """
    DPO loss for a single (prompt, chosen, rejected) triplet.
    
    Intuition: increase the probability of chosen relative to rejected,
    but don't move too far from the reference policy (controlled by beta).
    
    Mathematically equivalent to RLHF with an implicit reward model,
    but no RL training loop required.
    """
    import math
    # Log-ratio: how much more likely is chosen vs rejected, relative to reference?
    chosen_ratio  = log_prob_chosen_policy  - log_prob_chosen_ref
    rejected_ratio = log_prob_rejected_policy - log_prob_rejected_ref
    logit = beta * (chosen_ratio - rejected_ratio)
    # Binary cross-entropy loss on preference
    loss = -math.log(1 / (1 + math.exp(-logit)))   # -log sigmoid(logit)
    return loss
```

**DPO vs RLHF:**
- DPO: simpler, stable training, no reward model, widely used for chat models (Llama, Mistral)
- RLHF: more flexible (can use process reward models), better for math/reasoning tasks (o1-style)

---

## 9. Inference optimization

### Quantization

Reduce model weight precision to reduce memory and increase throughput:

| Format | Memory (7B model) | Quality loss | Use case |
|--------|-------------------|-------------|----------|
| float32 | 28 GB | None | Training |
| float16 / bfloat16 | 14 GB | Minimal | Standard inference |
| int8 (LLM.int8) | 7 GB | Small | GPU memory constrained |
| int4 (GPTQ, AWQ) | 3.5 GB | Moderate | Consumer GPUs |
| 2-bit | 1.75 GB | Significant | Edge/mobile |

### Continuous batching

Without continuous batching: a new request must wait until the current batch finishes. With continuous batching (vLLM): new requests join mid-flight as other sequences complete. Increases GPU utilization from ~30% to ~80%.

### Speculative decoding

Use a small "draft" model to propose K tokens. The large "verifier" model checks all K tokens in one forward pass. Accept greedily up to the first mismatch. Gives 2-4x speedup with identical output distribution.

```
Draft model (e.g., 3B):  generate tokens [t1, t2, t3, t4, t5] autoregressively
Verifier model (e.g., 70B): check all 5 in one pass
                            accept [t1, t2, t3], reject t4
                            sample correct t4 from verifier
Net: 4 tokens in the time it would take to generate 1 from verifier alone
```

---

## 10. Interview questions

**Architecture questions:**
- Why do we scale by √d_k in attention? (Prevent dot products from growing large with d_k, causing softmax saturation and vanishing gradients)
- What is the KV cache and why does it matter? (Avoid recomputing K and V for each generated token — reduces generation from O(n²) to O(n) per token)
- Why use multi-head attention instead of single-head? (Each head can specialize to attend to different types of relationships)
- What is the difference between encoder-only, decoder-only, and encoder-decoder transformers? (BERT vs. GPT vs. T5 — use case drives architecture choice)

**Decoding questions:**
- When would you use greedy vs. top-p vs. beam search? (Code → greedy; chat → top-p; translation → beam)
- What does temperature do to the output distribution? (T<1 sharpens, T>1 flattens, T→0 = greedy, T→∞ = uniform)
- Why is beam search often not used for creative tasks? (Tends to produce generic, repetitive output — sampling introduces diversity)

**Fine-tuning questions:**
- What is catastrophic forgetting and how does LoRA mitigate it? (Full FT overwrites existing weights; LoRA adds delta matrices, preserving base weights)
- What is the difference between RLHF and DPO? (RLHF trains explicit reward model + uses RL; DPO directly optimizes on preferences with a simpler loss)
- When would you choose SFT over RLHF? (SFT for format/style; RLHF/DPO when you need preference optimization for quality or safety)

**Systems questions:**
- How does KV cache memory scale with sequence length and batch size? (Linearly with both — the primary memory constraint for long-context serving)
- What is PagedAttention and why does it improve throughput? (Manages KV cache like virtual memory — eliminates fragmentation, enables more concurrent sequences)
- How does speculative decoding achieve speedup without changing output distribution? (Draft model proposes; verifier accepts or rejects atomically — mathematically equivalent to sampling from verifier)

---

{: .highlight }
**Previous:** [ML Spine Ch A — RAG in Depth →]({% link docs/ml-spine/chapter-rag.md %})
&nbsp;·&nbsp;
**Next:** [ML Spine Ch C — Agents in Depth →]({% link docs/ml-spine/chapter-agents.md %})
