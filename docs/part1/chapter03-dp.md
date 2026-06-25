---
layout: default
title: "Ch 3 · Dynamic Programming — Beam Search, Viterbi, RL"
parent: "Part 1 — Algorithm Spine"
nav_order: 3
description: "1D DP → 2D DP → beam search → Viterbi → RL value iteration → Tree of Thoughts."
---

# Dynamic Programming — Beam Search, Viterbi, RL
{: .no_toc }

*Chapter 3 · Part 1 — Algorithm Spine*
{: .text-grey-dk-000 .fs-4 }

<div class="reading-meta">
  ⏱ 40 min read &nbsp;·&nbsp;
  🟦 Engineering: L4 &nbsp;·&nbsp;
  🟩 ML: L4
</div>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mldsa/mldsa.github.io/blob/main/notebooks/chapter03_dp.ipynb)
&nbsp;
[![View Notebook](https://img.shields.io/badge/Notebook-View%20on%20GitHub-lightgrey?logo=github)](https://github.com/mldsa/mldsa.github.io/blob/main/notebooks/chapter03_dp.ipynb)

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Why this topic matters

Dynamic programming is the algorithm behind more ML systems than most ML engineers realize.

The LLM generating your next token? DP (beam search). The tokenizer splitting your text into subwords? DP (BPE segmentation). The NER model labeling entities in a sentence? DP (Viterbi). The RL agent planning the optimal action sequence? DP (Bellman equations, value iteration). The spell checker finding the closest word? DP (edit distance). BLEU/ROUGE evaluating your LLM output? DP (LCS, edit distance).

| DP Pattern | ML System | Connection |
|-----------|-----------|------------|
| 1D path DP | Beam search | State = partial sequence; transition = next token |
| Word break | BPE tokenization | Segment string into valid vocabulary tokens |
| Coin change | RL value iteration | Minimize steps/cost to reach goal state |
| Edit distance | BLEU, ROUGE, fuzzy match | Sequence similarity for LLM evaluation |
| LCS | Diff generation, code review | Find shared subsequence between outputs |
| Viterbi | NER, POS tagging, CRF decoding | Best label sequence over observed tokens |

**For the engineering-inclined reader:** DP unlocks the hardest LeetCode mediums and most hards. This chapter gives you the pattern and shows where it appears in systems you'll design.

**For the ML-inclined reader:** Every time you run `model.generate()`, a DP algorithm is choosing the output sequence. This chapter shows you the math inside beam search, why Viterbi works, and how RL planning connects to the coin change problem.

---

## 2. Engineering view

### What makes a problem DP-solvable

A problem can be solved with DP if it has two properties:

**1. Optimal substructure:** The optimal solution to the whole problem contains optimal solutions to subproblems.

**2. Overlapping subproblems:** The same subproblems appear repeatedly if you solve naively by recursion.

If both hold, DP turns exponential brute-force into polynomial time by storing and reusing results.

### Two implementation styles

**Top-down (memoization):** Write the natural recursion. Cache results.

```python
from functools import lru_cache

def fib(n):
    @lru_cache(maxsize=None)
    def dp(i):
        if i <= 1: return i
        return dp(i-1) + dp(i-2)
    return dp(n)
```

**Bottom-up (tabulation):** Fill a table from base cases up to the answer. No recursion overhead.

```python
def fib(n):
    if n <= 1: return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

### The DP design recipe

Every DP solution follows the same four steps:

1. **Define the state:** What does `dp[i]` (or `dp[i][j]`) mean?
2. **Write the recurrence:** How does `dp[i]` depend on smaller states?
3. **Identify base cases:** What are the smallest states?
4. **Determine iteration order:** Which direction must I fill the table?

---

## 3. 1D DP patterns

### Most 1D DP falls into three templates

**Template A — Linear scan, take or skip:**
```
dp[i] = best(dp[i-1] + use(i),  dp[i-1] without i)
```
Examples: House Robber, Maximum Subarray, Stock prices

**Template B — Reachability / counting:**
```
dp[i] = sum of dp[j] for all valid j < i
```
Examples: Climbing Stairs, Coin Change, Decode Ways, Word Break

**Template C — Running optimum:**
```
dp[i] = best answer ending exactly at position i
```
Examples: Longest Increasing Subsequence, Maximum Subarray

### Climbing stairs (#70)

```python
def climb_stairs(n: int) -> int:
    """
    State:      dp[i] = number of ways to reach stair i
    Recurrence: dp[i] = dp[i-1] + dp[i-2]
    ML analog:  count valid decoding paths in beam search before pruning.
    """
    if n <= 2: return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        prev2, prev1 = prev1, prev1 + prev2
    return prev1
```

### Coin change (#322) — the RL connection

```python
def coin_change(coins: list[int], amount: int) -> int:
    """
    State:      dp[i] = min coins to make amount i
    Recurrence: dp[i] = min(dp[i - c] + 1) for each coin c

    RL CONNECTION — this IS value iteration:
      States: amounts 0..amount  |  Actions: coins (transitions)
      Bellman: V(s) = min over a of [cost(a) + V(s')]
               dp[i] = 1 + min(dp[i - c] for c in coins)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

### Word break (#139) — tokenization DP

```python
def word_break(s: str, word_dict: list[str]) -> bool:
    """
    State:      dp[i] = True if s[:i] can be segmented
    Recurrence: dp[i] = OR over j<i of (dp[j] AND s[j:i] in word_dict)

    TOKENIZATION: This is what a tokenizer does. BPE finds the
    *best* segmentation (fewest tokens), not just any valid one.
    """
    word_set = set(word_dict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[len(s)]
```

---

## 4. 2D DP patterns

### The "two strings → 2D table" signal

If a problem gives you **two strings** and asks for similarity, distance, alignment, or overlap — your first thought should be 2D DP.

```
      ""  r  o  s
  ""  [0, 1, 2, 3]
  h   [1, 1, 2, 3]
  o   [2, 2, 1, 2]
  r   [3, 2, 2, 2]
  s   [4, 3, 3, 2]  ← edit_distance("hors", "ros") = 2
  e   [5, 4, 4, 3]
```

### Edit distance (#72) — the sequence metric

```python
def edit_distance(word1: str, word2: str) -> int:
    """
    State:   dp[i][j] = edit distance between word1[:i] and word2[:j]
    Recurrence:
        if match:  dp[i][j] = dp[i-1][j-1]
        else:      dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    LLM EVALUATION: WER (Word Error Rate), spelling correction,
    output deduplication, agent fuzzy tool matching.
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def fuzzy_tool_match(query: str, tool_names: list[str]) -> str:
    """Find closest valid tool name when agent produces a typo."""
    return min(tool_names, key=lambda t: edit_distance(query.lower(), t.lower()))

# fuzzy_tool_match("searh_web", ["search_web", "run_python"]) → "search_web"
```

### LCS → ROUGE-L

```python
def lcs(text1: str, text2: str) -> int:
    """
    Longest Common Subsequence — characters need not be contiguous.
    Powers ROUGE-L, code similarity, plagiarism detection.
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 score using LCS. Standard summarization metric."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    lcs_len = lcs(" ".join(ref_tokens), " ".join(hyp_tokens))
    if not ref_tokens or not hyp_tokens:
        return 0.0
    precision = lcs_len / len(hyp_tokens)
    recall    = lcs_len / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

print(f"ROUGE-L: {rouge_l('the cat sat on the mat', 'the cat is on the mat'):.3f}")  # ~0.909
```

---

## 5. Beam search: DP for LLM decoding

Beam search is 1D sequence DP with bounded state space.

- **State:** `dp[t]` = set of K best partial sequences at time step t
- **Recurrence:** `dp[t+1]` = top-K over all `(dp[t] state + next_token)` expansions
- Compare to `word_break_all_paths` — that's beam search with `beam_width=∞`

```python
import heapq, math

def beam_search_dp(
    vocab_size: int,
    score_fn,           # (partial_sequence) → list of log_probs for each next token
    beam_width: int = 3,
    max_len: int = 10,
    bos_token: int = 0,
    eos_token: int = 1,
) -> list[tuple[float, list[int]]]:
    """
    dp[0]: single start state {(0.0, [BOS])}
    dp[t]: top beam_width states from expanding dp[t-1]
    Final: argmax over completed states
    """
    active_beams = [(0.0, [bos_token])]
    completed = []

    for t in range(max_len):
        if not active_beams:
            break
        candidates = []
        for neg_lp, seq in active_beams:
            if seq[-1] == eos_token:
                completed.append((neg_lp, seq))
                continue
            log_probs = score_fn(seq)
            for token_id, lp in enumerate(log_probs):
                candidates.append((neg_lp - lp, seq + [token_id]))
        active_beams = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])

    completed.extend(active_beams)
    completed.sort(key=lambda x: x[0])
    return [(math.exp(-neg_lp), seq) for neg_lp, seq in completed]
```

---

## 6. Viterbi: DP for sequence labeling

Viterbi is 2D DP for finding the most likely label sequence given observed tokens. It powers NER, POS tagging, and CRF decoding.

- **State:** `viterbi[t][s]` = log prob of most likely path ending in state s at time t
- **Recurrence:** `viterbi[t][s] = log(emission[s][obs[t]]) + max over s' of (viterbi[t-1][s'] + log(transition[s'][s]))`

```python
import math

def viterbi(
    observations: list[int],
    n_states: int,
    transition: list[list[float]],   # P(s'|s)
    emission: list[list[float]],     # P(o|s)
    initial: list[float],            # P(s at t=0)
) -> list[int]:
    """
    NER EXAMPLE: States = B-PER, I-PER, B-ORG, I-ORG, O
    Viterbi finds the globally optimal label sequence.
    Greedy labeling cannot — it commits at each step.

    Viterbi is beam search with beam_width=n_states and exact scores.
    Beam search is approximate Viterbi over a large vocab with pruning.
    """
    T, S = len(observations), n_states
    dp = [[float('-inf')] * S for _ in range(T)]
    backptr = [[0] * S for _ in range(T)]

    for s in range(S):
        if initial[s] > 0 and emission[s][observations[0]] > 0:
            dp[0][s] = math.log(initial[s]) + math.log(emission[s][observations[0]])

    for t in range(1, T):
        for s in range(S):
            if emission[s][observations[t]] <= 0:
                continue
            emit_lp = math.log(emission[s][observations[t]])
            best_prev, best_lp = 0, float('-inf')
            for prev_s in range(S):
                if dp[t-1][prev_s] == float('-inf') or transition[prev_s][s] <= 0:
                    continue
                lp = dp[t-1][prev_s] + math.log(transition[prev_s][s])
                if lp > best_lp:
                    best_lp, best_prev = lp, prev_s
            dp[t][s] = best_lp + emit_lp
            backptr[t][s] = best_prev

    path = [0] * T
    path[T-1] = max(range(S), key=lambda s: dp[T-1][s])
    for t in range(T-2, -1, -1):
        path[t] = backptr[t+1][path[t+1]]
    return path
```

---

## 7. RL value iteration: DP as planning

Value iteration is DP for reinforcement learning, directly applying Bellman's principle:

```
V*(s) = max over a of [R(s,a) + γ · V*(s')]
```

This is the exact same recurrence as Coin Change: `dp[i] = min over coins of [1 + dp[i - coin]]`. Coin Change minimizes cost. Value iteration maximizes reward. Same DP, opposite optimization.

```python
def value_iteration(
    n_states: int,
    n_actions: int,
    transitions,   # transitions[s][a] = (next_state, reward)
    gamma: float = 0.9,
    threshold: float = 1e-6,
    max_iter: int = 1000,
) -> tuple[list[float], list[int]]:
    """
    AGENT PLANNING CONNECTION:
    - States: agent's world states (task progress, tool results)
    - Actions: tools or reasoning steps available
    - V*(s): expected return from state s under optimal policy
    - π*(s): optimal action to take in state s
    """
    V = [0.0] * n_states
    for _ in range(max_iter):
        delta = 0.0
        new_V = V[:]
        for s in range(n_states):
            action_values = [transitions[s][a][1] + gamma * V[transitions[s][a][0]]
                             for a in range(n_actions)]
            new_V[s] = max(action_values)
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < threshold:
            break
    policy = [max(range(n_actions),
                  key=lambda a: transitions[s][a][1] + gamma * V[transitions[s][a][0]])
              for s in range(n_states)]
    return V, policy
```

---

## 8. Tree of Thoughts: beam search over reasoning

```python
def best_reasoning_path_dp(
    initial_thought: str,
    expand_fn,     # (thought) → list of next thoughts
    score_fn,      # (thought) → float
    beam_width: int = 3,
    depth: int = 4,
) -> tuple[float, list[str]]:
    """
    Tree of Thoughts with beam search (DP + pruning).
    This is beam_search_dp applied to thoughts instead of tokens.
    Used in o1-style reasoning: score intermediate steps, prune weak chains.
    """
    import heapq
    beams = [(0.0, [initial_thought])]

    for _ in range(depth):
        candidates = []
        for neg_score, path in beams:
            for next_thought in expand_fn(path[-1]):
                step_score = score_fn(next_thought)
                new_score = neg_score - math.log(step_score + 1e-10)
                candidates.append((new_score, path + [next_thought]))
        beams = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])

    best_neg_score, best_path = beams[0]
    return math.exp(-best_neg_score), best_path
```

---

## 9. Complexity analysis

| Problem | Time | Space | Reducible? |
|---------|------|-------|------------|
| Climbing Stairs | O(n) | O(1) | 2 vars |
| House Robber | O(n) | O(1) | 2 vars |
| Coin Change | O(n × k) | O(n) | No |
| Word Break | O(n² × m) | O(n) | No |
| LIS (DP) | O(n²) | O(n) | — |
| LIS (binary search) | O(n log n) | O(n) | — |
| Edit Distance | O(m × n) | O(n) | 1 row |
| LCS | O(m × n) | O(n) | 1 row |
| Viterbi | O(T × S²) | O(T × S) | — |
| Beam Search | O(T × K × V) | O(K × T) | — |
| Value Iteration | O(I × S × A) | O(S) | — |

---

## 10. Interview patterns

| You see... | DP state | Recurrence pattern |
|-----------|---------|-------------------|
| Count ways to reach target | `dp[i]` = # ways | `dp[i] += dp[i - option]` |
| Minimum cost to reach target | `dp[i]` = min cost | `dp[i] = min(dp[i-c] + 1)` |
| Can we reach target? | `dp[i]` = bool | `dp[i] = OR(dp[j] for valid j)` |
| Best subsequence | `dp[i]` = best ending at i | `dp[i] = max(dp[j] + val)` |
| Two sequences | `dp[i][j]` | depends on match/mismatch |
| Sequence segmentation | `dp[i]` = segmentable | `dp[i] = OR(dp[j] if s[j:i] valid)` |

---

## 11. Practice problems

| # | Problem | Pattern | Difficulty |
|---|---------|---------|-----------|
| 70 | Climbing Stairs | Counting paths | Easy |
| 198 | House Robber | Take/skip | Medium |
| 322 | Coin Change | Min cost | Medium |
| 139 | Word Break | Segmentation | Medium |
| 300 | LIS | Running optimum | Medium |
| 72 | Edit Distance | 2D sequence | Hard |
| 1143 | LCS | 2D sequence | Medium |
| 5 | Longest Palindromic Substring | 2D interval | Medium |
| 91 | Decode Ways | Counting + constraints | Medium |

**ML problems:** Implement ROUGE-L from scratch. Build a spell corrector using edit distance over a vocabulary. Implement Viterbi for a toy NER task. Implement beam search for a character-level language model at beam_width 1, 3, and 10.

**Theory questions:** Why does beam search sometimes produce worse outputs than sampling for creative tasks? Edit distance is symmetric. BLEU is not. Why? Why does Viterbi find the globally optimal label sequence while greedy labeling doesn't?

---

## 12. Common mistakes

**LeetCode:** Defining the wrong state. Off-by-one errors in base cases. Iterating in the wrong direction. Using O(n²) space when O(n) is possible with a rolling array.

**ML systems:** Confusing ROUGE-L (LCS-based) with ROUGE-N (n-gram overlap). Not length-normalizing log probabilities in beam search (shorter sequences win unfairly). In value iteration: stopping before convergence gives suboptimal policies.

---

{: .highlight }
**Previous:** [Ch 2 — Top-K, Vector Search, and RAG Retrieval →]({% link docs/part1/chapter02-topk.md %})
