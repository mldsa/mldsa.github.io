---
layout: default
title: "ML-C · Agents in Depth"
parent: "ML Spine"
nav_order: 3
description: "LLM agents end to end: ReAct loop, tool use, memory systems, planning, evaluation, and production failure modes."
---

# Agents in Depth
{: .no_toc }

*ML Spine · Chapter C*
{: .text-grey-dk-000 .fs-4 }

<div class="reading-meta">
  ⏱ 35 min read &nbsp;·&nbsp;
  🟩 ML: L3–L4 &nbsp;·&nbsp;
  🟦 DSA inside: HashMap · Graph BFS · DP · LRU Cache
</div>

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. What an agent is

An **LLM agent** is a system where an LLM controls a loop: it observes the environment, reasons about what to do, executes an action, observes the result, and repeats until the task is done.

The minimal agent loop:

```
while not done:
    observation = get_observation()     # what can I see?
    thought = llm.reason(observation)   # what should I do?
    action = llm.choose_action(thought) # pick a tool/action
    result = execute(action)            # run it
    update_memory(thought, action, result)
```

What distinguishes an agent from a simple LLM call:
- **Tools:** the LLM can call external functions (search, code execution, APIs, databases)
- **Memory:** the agent accumulates context across steps
- **Planning:** the agent decides what to do next based on what it knows so far
- **Loop:** the agent runs multiple steps, not just one

---

## 2. ReAct: the canonical agent pattern

ReAct (Reasoning + Acting) is the foundational paper (Yao et al. 2022) that established the standard agent structure. The LLM interleaves **thought** and **action** in its output:

```
Thought: I need to find the population of Tokyo.
Action: search("Tokyo population 2024")
Observation: Tokyo metropolitan area population is approximately 37.4 million.
Thought: Now I have the population. I can answer the question.
Action: finish("Tokyo's population is approximately 37.4 million.")
```

### ReAct loop implementation

```python
class ReActAgent:
    """
    Minimal ReAct agent. LLM generates Thought/Action pairs.
    Tool executor runs the action and returns the observation.
    Loop continues until LLM calls finish() or max_steps reached.
    
    DSA inside: graph state machine (nodes = states, edges = actions).
    """
    def __init__(self, llm_fn, tools: dict, max_steps: int = 10, verbose: bool = True):
        self.llm = llm_fn          # (prompt: str) → str
        self.tools = tools         # {"tool_name": callable}
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, task: str) -> str:
        history = []
        system = self._build_system_prompt()

        for step in range(self.max_steps):
            # Build prompt from task + history
            prompt = self._build_prompt(system, task, history)
            # LLM generates next thought + action
            response = self.llm(prompt)
            thought, action, action_input = self._parse_response(response)

            if self.verbose:
                print(f"[Step {step+1}] Thought: {thought}")
                print(f"           Action: {action}({action_input})")

            history.append({"thought": thought, "action": action, "input": action_input})

            # Terminal action
            if action == "finish":
                return action_input

            # Execute tool
            if action not in self.tools:
                observation = f"Error: tool '{action}' not found. Available: {list(self.tools.keys())}"
            else:
                try:
                    observation = str(self.tools[action](action_input))
                except Exception as e:
                    observation = f"Error executing {action}: {e}"

            if self.verbose:
                print(f"           Observation: {observation[:200]}")

            history[-1]["observation"] = observation

        return "Max steps reached without finishing."

    def _build_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            f"- {name}: {fn.__doc__ or 'No description'}"
            for name, fn in self.tools.items()
        )
        return f"""You are an agent that solves tasks step by step.

Available tools:
{tool_descriptions}
- finish(answer): Return the final answer.

Format your response as:
Thought: <your reasoning>
Action: <tool_name>
Input: <tool input>"""

    def _build_prompt(self, system: str, task: str, history: list) -> str:
        prompt = f"{system}\n\nTask: {task}\n\n"
        for step in history:
            prompt += f"Thought: {step['thought']}\nAction: {step['action']}\nInput: {step['input']}\n"
            if "observation" in step:
                prompt += f"Observation: {step['observation']}\n\n"
        return prompt

    def _parse_response(self, response: str) -> tuple[str, str, str]:
        """Parse LLM output into (thought, action, input)."""
        lines = response.strip().split('\n')
        thought = action = action_input = ""
        for line in lines:
            if line.startswith("Thought:"):
                thought = line[len("Thought:"):].strip()
            elif line.startswith("Action:"):
                action = line[len("Action:"):].strip()
            elif line.startswith("Input:"):
                action_input = line[len("Input:"):].strip()
        return thought, action, action_input
```

---

## 3. Tool use and function calling

### Tool registry

The agent's tools are a HashMap: tool name → callable. See [Ch 1]({% link docs/part1/chapter01-hashing.md %}) for the HashMap implementation.

```python
class ToolRegistry:
    """
    Central registry for agent tools.
    DSA inside: HashMap with fuzzy fallback (edit distance, Ch 3).
    """
    def __init__(self):
        self._tools: dict[str, callable] = {}
        self._schemas: dict[str, dict] = {}

    def register(self, name: str, fn: callable, schema: dict = None):
        """Register a tool with its callable and JSON schema."""
        self._tools[name] = fn
        self._schemas[name] = schema or {"description": fn.__doc__ or ""}

    def call(self, name: str, *args, **kwargs):
        if name not in self._tools:
            # Fuzzy fallback: find closest tool name
            closest = self._fuzzy_match(name)
            raise KeyError(f"Tool '{name}' not found. Did you mean '{closest}'?")
        return self._tools[name](*args, **kwargs)

    def _fuzzy_match(self, query: str) -> str:
        """Edit distance fallback when tool name is malformed."""
        from docs.part1.chapter03_dp import edit_distance  # conceptual import
        return min(self._tools.keys(), key=lambda t: edit_distance(query, t))

    def get_schema_prompt(self) -> str:
        """Format tools for LLM system prompt."""
        lines = []
        for name, schema in self._schemas.items():
            lines.append(f"- {name}: {schema.get('description', '')}")
        return "\n".join(lines)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


# Example tool implementations
def web_search(query: str) -> str:
    """Search the web and return top results as text."""
    # In production: call Serper, Tavily, or Bing Search API
    return f"[Search results for '{query}']"

def run_python(code: str) -> str:
    """Execute Python code in a sandboxed environment and return output."""
    # In production: use subprocess with timeout, or e2b.dev
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {})
        return buf.getvalue() or "Code executed successfully (no output)."
    except Exception as e:
        return f"Error: {e}"

def read_file(path: str) -> str:
    """Read a file and return its contents."""
    try:
        with open(path) as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


registry = ToolRegistry()
registry.register("search", web_search)
registry.register("python", run_python)
registry.register("read_file", read_file)
```

### Structured tool calling (OpenAI function calling)

Modern LLM APIs support structured tool calling — the LLM outputs a JSON object specifying the tool and arguments, rather than free text:

```python
def structured_tool_call(
    llm_client,
    messages: list[dict],
    tools: list[dict],   # JSON schemas
    registry: ToolRegistry,
    max_rounds: int = 10,
) -> str:
    """
    OpenAI-style function calling loop.
    LLM outputs tool_call objects; we execute and append results.
    """
    for _ in range(max_rounds):
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            # No tool call → final answer
            return msg.content

        # Execute all tool calls in this turn
        for tc in msg.tool_calls:
            result = registry.call(tc.function.name, **json.loads(tc.function.arguments))
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    return "Max rounds reached."
```

---

## 4. Memory systems

Agents have four types of memory, analogous to computer memory architecture:

| Memory Type | Analogy | Implementation | Lifespan |
|-------------|---------|----------------|---------|
| In-context (working) | RAM | Messages list / context window | Current session |
| External semantic | Hard drive | Vector store | Persistent |
| External episodic | Log file | Key-value store / DB | Persistent |
| Parametric | ROM | Model weights | Training |

### In-context memory (working memory)

The conversation history in the context window. Limited by context size. The agent sees everything here without retrieval.

```python
class ConversationBuffer:
    """
    Simple FIFO buffer for conversation history.
    When buffer exceeds max_tokens, oldest messages are dropped.
    DSA inside: deque with O(1) append and O(1) popleft.
    """
    def __init__(self, max_tokens: int = 4000):
        from collections import deque
        self.messages = deque()
        self.max_tokens = max_tokens
        self.current_tokens = 0

    def add(self, role: str, content: str):
        token_count = len(content.split())  # rough estimate
        self.messages.append({"role": role, "content": content, "tokens": token_count})
        self.current_tokens += token_count
        # Evict oldest until within budget
        while self.current_tokens > self.max_tokens and len(self.messages) > 1:
            evicted = self.messages.popleft()
            self.current_tokens -= evicted["tokens"]

    def get_messages(self) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]
```

### Semantic memory (long-term vector store)

Store important information as embeddings. Retrieve by similarity when needed. This is RAG applied to agent memory.

```python
class SemanticMemory:
    """
    Long-term memory backed by vector search.
    Agent stores important facts/observations; retrieves relevant ones per query.
    DSA inside: vector search (Ch 2 + ML Spine Ch A).
    """
    def __init__(self, embed_fn, index):
        self.embed_fn = embed_fn
        self.index = index
        self.memories: list[str] = []

    def store(self, content: str, metadata: dict = None):
        embedding = self.embed_fn(content)
        self.index.add(embedding, {"text": content, **(metadata or {})})
        self.memories.append(content)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        q_emb = self.embed_fn(query)
        results = self.index.search(q_emb, k=k)
        return [meta["text"] for meta, _ in results]

    def summarize_and_compress(self, llm_fn, keep_last_n: int = 5):
        """
        Compress old memories into a summary to manage memory size.
        Pattern: summarize the older half, keep recent N verbatim.
        """
        if len(self.memories) <= keep_last_n:
            return
        old = self.memories[:-keep_last_n]
        summary = llm_fn(f"Summarize these memories concisely:\n" + "\n".join(old))
        self.memories = [summary] + self.memories[-keep_last_n:]
```

### Episodic memory (experience replay)

Store past task executions as (task, trajectory, outcome) tuples. Retrieve similar past experiences to guide current planning:

```python
class EpisodicMemory:
    """
    Store past agent trajectories. Retrieve similar experiences for few-shot planning.
    
    Example: agent solved "summarize PDF and email to team" before.
    When given similar task, retrieve that episode as a few-shot example.
    """
    def __init__(self, embed_fn, index):
        self.embed_fn = embed_fn
        self.index = index

    def store_episode(self, task: str, trajectory: list[dict], outcome: str, success: bool):
        embedding = self.embed_fn(task)
        self.index.add(embedding, {
            "task": task,
            "trajectory": trajectory,
            "outcome": outcome,
            "success": success,
        })

    def retrieve_similar(self, task: str, k: int = 3) -> list[dict]:
        """Get K most similar past episodes, prioritizing successful ones."""
        results = self.index.search(self.embed_fn(task), k=k*2)
        # Prefer successful episodes
        episodes = [meta for meta, _ in results]
        episodes.sort(key=lambda e: (not e["success"], 0))
        return episodes[:k]
```

---

## 5. Planning

### The planning problem

For multi-step tasks, the agent must decide: what sequence of actions gets from current state to goal? This is the core planning problem — and it's exactly what DP and search algorithms solve.

### Single-agent sequential planning (chain)

The simplest plan: execute steps one by one, each informed by the previous:

```python
class SequentialPlanner:
    """
    Plan: LLM generates a numbered list of steps upfront.
    Execute: run each step, updating context as we go.
    Good for: predictable, well-structured tasks.
    Bad for: tasks requiring dynamic replanning.
    """
    def __init__(self, agent: ReActAgent, llm_fn):
        self.agent = agent
        self.llm = llm_fn

    def plan_and_execute(self, task: str) -> str:
        # Step 1: Generate plan
        plan_prompt = f"Break this task into numbered steps:\n\nTask: {task}\n\nSteps:"
        plan_text = self.llm(plan_prompt)
        steps = self._parse_steps(plan_text)

        # Step 2: Execute each step
        context = f"Overall task: {task}\n\n"
        results = []
        for i, step in enumerate(steps, 1):
            result = self.agent.run(f"{context}Current step {i}: {step}")
            results.append(result)
            context += f"Step {i} result: {result}\n"

        # Step 3: Synthesize
        synthesis_prompt = f"Task: {task}\n\nResults:\n" + "\n".join(
            f"{i+1}. {r}" for i, r in enumerate(results)
        ) + "\n\nFinal answer:"
        return self.llm(synthesis_prompt)

    def _parse_steps(self, text: str) -> list[str]:
        lines = text.strip().split('\n')
        return [l.split('.', 1)[-1].strip() for l in lines if l.strip() and l[0].isdigit()]
```

### Tree of Thoughts planning

When multiple solution paths are possible, Tree of Thoughts (ToT) generates several candidate next steps, scores them, and follows the best path — beam search over reasoning:

```python
class TreeOfThoughtsPlanner:
    """
    Generate K candidate next thoughts, score each, keep best beam_width.
    DSA inside: beam search DP (Ch 3).
    Used in: math problem solving, multi-step research, code generation.
    """
    def __init__(self, llm_fn, score_fn, beam_width: int = 3, max_depth: int = 5):
        self.llm = llm_fn
        self.score = score_fn   # (task, thought_path) → float
        self.beam_width = beam_width
        self.max_depth = max_depth

    def solve(self, task: str) -> list[str]:
        # Each beam: list of thoughts (a reasoning path)
        beams = [[]]

        for depth in range(self.max_depth):
            candidates = []
            for path in beams:
                # Generate K candidate next thoughts for this path
                context = f"Task: {task}\nThoughts so far:\n" + "\n".join(path)
                next_thoughts = self._generate_candidates(context, k=3)
                for thought in next_thoughts:
                    new_path = path + [thought]
                    s = self.score(task, new_path)
                    candidates.append((s, new_path))

            # Keep top beam_width by score
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = [path for _, path in candidates[:self.beam_width]]

            # Check if any beam reached a solution
            for path in beams:
                if self._is_solution(path[-1]):
                    return path

        return beams[0] if beams else []

    def _generate_candidates(self, context: str, k: int) -> list[str]:
        """Generate K diverse candidate next thoughts."""
        prompt = f"{context}\n\nGenerate {k} different possible next reasoning steps (one per line):"
        response = self.llm(prompt)
        return [l.strip() for l in response.split('\n') if l.strip()][:k]

    def _is_solution(self, thought: str) -> bool:
        return "answer:" in thought.lower() or "final:" in thought.lower()
```

### DAG-based task decomposition

Complex tasks decompose into subtasks with dependencies — a directed acyclic graph. Execute in topological order:

```python
from collections import deque

class DAGTaskPlanner:
    """
    Decompose complex task into DAG of subtasks.
    Execute respecting dependencies (topological sort).
    DSA inside: topological sort BFS (Kahn's algorithm).
    
    Use case: "research competitors, summarize findings, draft report, send email"
    Dependencies: draft_report depends on summarize_findings depends on research
    """
    def __init__(self, agent: ReActAgent, llm_fn):
        self.agent = agent
        self.llm = llm_fn

    def plan_dag(self, task: str) -> dict:
        """Ask LLM to decompose task into subtasks with dependencies."""
        prompt = f"""Decompose this task into subtasks with dependencies.
Output as JSON: {{"subtasks": [{{"id": "t1", "task": "...", "depends_on": []}}]}}

Task: {task}"""
        response = self.llm(prompt)
        import json, re
        match = re.search(r'\{.*\}', response, re.DOTALL)
        return json.loads(match.group()) if match else {"subtasks": []}

    def execute_dag(self, task: str) -> dict[str, str]:
        """Execute subtasks in topological order, passing results forward."""
        plan = self.plan_dag(task)
        subtasks = {s["id"]: s for s in plan["subtasks"]}
        in_degree = {sid: len(s["depends_on"]) for sid, s in subtasks.items()}
        results: dict[str, str] = {}

        # Kahn's algorithm for topological sort + execution
        queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
        while queue:
            sid = queue.popleft()
            st = subtasks[sid]
            # Build context from dependency results
            dep_context = "\n".join(f"{d}: {results[d]}" for d in st["depends_on"])
            prompt = f"Task: {st['task']}\n\nContext from previous steps:\n{dep_context}"
            results[sid] = self.agent.run(prompt)
            # Unblock dependent tasks
            for other_id, other in subtasks.items():
                if sid in other["depends_on"]:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        return results
```

---

## 6. Multi-agent systems

### Why multiple agents

A single agent with all tools in its context gets confused on complex tasks. Specialized agents — each with a focused toolset and prompt — perform better. A coordinator routes tasks to specialists.

```python
class MultiAgentSystem:
    """
    Coordinator + specialist agents pattern.
    DSA inside: graph routing (HashMap of agent_name → agent).
    """
    def __init__(self, coordinator_llm, specialists: dict[str, ReActAgent]):
        self.coordinator = coordinator_llm
        self.specialists = specialists   # {"researcher": agent1, "coder": agent2, ...}

    def run(self, task: str) -> str:
        # Coordinator decides which specialist to use
        routing_prompt = f"""Route this task to the best specialist.
Specialists: {list(self.specialists.keys())}
Task: {task}
Respond with: specialist_name|subtask_for_specialist"""
        routing = self.coordinator(routing_prompt)
        specialist_name, subtask = routing.split("|", 1)
        specialist_name = specialist_name.strip()

        if specialist_name not in self.specialists:
            return f"Unknown specialist: {specialist_name}"

        result = self.specialists[specialist_name].run(subtask.strip())

        # Coordinator synthesizes
        synthesis = self.coordinator(
            f"Task: {task}\nSpecialist result: {result}\nSynthesize final answer:"
        )
        return synthesis
```

---

## 7. Agent evaluation

Agents are hard to evaluate because:
- The task space is open-ended
- The same task can succeed via different trajectories
- Intermediate steps may be wrong even if the final answer is right

### Trajectory evaluation

```python
def evaluate_trajectory(
    trajectory: list[dict],   # [{"thought": ..., "action": ..., "observation": ...}]
    task: str,
    correct_answer: str,
    score_fn,                 # (task, answer) → float (LLM judge)
) -> dict:
    """
    Evaluate an agent's execution trace.
    """
    # 1. Task completion
    final_action = trajectory[-1] if trajectory else {}
    final_answer = final_action.get("input", "") if final_action.get("action") == "finish" else ""
    task_score = score_fn(task, final_answer)

    # 2. Efficiency (fewer steps is better for same quality)
    n_steps = len(trajectory)

    # 3. Tool correctness (were tools called with valid inputs?)
    tool_errors = sum(
        1 for step in trajectory
        if "Error" in step.get("observation", "")
    )

    # 4. Reasoning quality (do thoughts logically lead to actions?)
    reasoning_prompt = f"""Rate the reasoning quality 0-1.
Task: {task}
Trajectory:
{format_trajectory(trajectory)}
Score (0-1):"""
    reasoning_score = float(score_fn("", score_fn(reasoning_prompt, "")))

    return {
        "task_completion": task_score,
        "steps": n_steps,
        "tool_error_rate": tool_errors / max(n_steps, 1),
        "reasoning_quality": reasoning_score,
        "final_answer": final_answer,
    }


def format_trajectory(trajectory: list[dict]) -> str:
    lines = []
    for i, step in enumerate(trajectory, 1):
        lines.append(f"Step {i}:")
        lines.append(f"  Thought: {step.get('thought', '')}")
        lines.append(f"  Action: {step.get('action', '')}({step.get('input', '')})")
        lines.append(f"  Observation: {step.get('observation', '')[:100]}")
    return "\n".join(lines)
```

### Benchmark frameworks

| Framework | What it tests | URL |
|-----------|--------------|-----|
| **GAIA** | Real-world reasoning (files, web, math) | Meta / HuggingFace |
| **SWE-bench** | GitHub issue resolution (code agents) | Princeton NLP |
| **WebArena** | Web navigation + task completion | CMU |
| **AgentBench** | 8 environments: OS, DB, web, games | Tsinghua |
| **ToolBench** | Tool use over 16k+ real APIs | Tsinghua |

A good agent on GAIA Level 1 scores >70%. GPT-4 baseline without scaffolding: ~36%.

---

## 8. Production failure modes

### The most common ways agents break

**1. Tool calling hallucination** — LLM invents tool names or arguments that don't exist.

Fix: strict JSON schema validation + fuzzy matching fallback + "available tools" prominently in system prompt.

**2. Context window overflow** — long trajectories fill the context, causing the LLM to forget the original task.

Fix: summarize older turns periodically; use semantic memory for important facts; set hard step limits.

**3. Infinite loops** — agent gets stuck repeating the same action because the observation doesn't change.

Fix: track (action, input) pairs; abort if the same call appears 3+ times.

```python
class LoopDetector:
    def __init__(self, max_repeats: int = 3):
        self.seen = {}
        self.max_repeats = max_repeats

    def check(self, action: str, action_input: str) -> bool:
        key = (action, action_input[:100])
        self.seen[key] = self.seen.get(key, 0) + 1
        return self.seen[key] >= self.max_repeats   # True = loop detected
```

**4. Tool result trust** — agent believes incorrect tool outputs (e.g., a search result that's wrong or a code execution that had a silent error).

Fix: validate tool outputs where possible; ask the LLM to explicitly check results before proceeding.

**5. Prompt injection** — malicious content in tool outputs tries to hijack the agent's instructions.

Fix: treat tool output as untrusted data; don't interpolate raw tool output into system prompts; use output sanitization.

**6. Cost/latency explosion** — agents can make hundreds of LLM calls on complex tasks.

Fix: set token budgets per task; cache tool results; use smaller models for cheaper steps (routing); set max_steps limits.

### Production agent checklist

```python
class ProductionAgent:
    """
    Hardened production agent with standard safeguards.
    """
    def __init__(self, llm_fn, tools, max_steps=15, max_tokens_per_run=50_000):
        self.agent = ReActAgent(llm_fn, tools, max_steps=max_steps)
        self.loop_detector = LoopDetector(max_repeats=3)
        self.token_budget = max_tokens_per_run
        self.tokens_used = 0

    def run(self, task: str) -> dict:
        start_time = __import__("time").time()
        try:
            result = self.agent.run(task)
            return {
                "success": True,
                "result": result,
                "steps": len(self.agent.history) if hasattr(self.agent, "history") else -1,
                "latency_s": __import__("time").time() - start_time,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "latency_s": __import__("time").time() - start_time}
```

---

## 9. Agent patterns reference

| Pattern | When to use | Key component |
|---------|------------|---------------|
| **ReAct** | Single-task, single agent | Thought/Action/Observation loop |
| **Plan-and-Execute** | Predictable multi-step tasks | Upfront plan, then execute |
| **Tree of Thoughts** | Tasks with multiple solution paths | Beam search over thoughts |
| **Reflection** | Tasks needing self-correction | Agent critiques own output |
| **Multi-agent (specialist)** | Complex tasks with distinct subtasks | Coordinator + specialists |
| **DAG decomposition** | Tasks with dependency structure | Topological sort execution |
| **RAG agent** | Knowledge-intensive tasks | Agent + retrieval memory |
| **Code interpreter** | Math, data analysis, automation | Agent + Python execution |

---

## 10. Interview questions

**Architecture questions:**
- What is the ReAct pattern and why does it outperform pure chain-of-thought? (Interleaving action-observation grounds reasoning in real results; CoT can hallucinate without grounding)
- How does an agent decide which tool to use? (LLM trained on tool descriptions + function calling schema; few-shot examples in system prompt)
- What is the difference between in-context and external memory? (In-context is fast but limited and ephemeral; external is persistent but requires retrieval)

**Planning questions:**
- When would you use Tree of Thoughts vs. sequential planning? (ToT for tasks with multiple solution paths and a verifiable scoring function; sequential for linear workflows)
- How does multi-agent coordination differ from a single agent with many tools? (Specialization reduces context pollution; parallel execution possible; clearer separation of concerns)
- What is the connection between agent planning and DP? (Both find optimal sequences of decisions; ToT = beam search DP; DAG execution = topological sort of a value function)

**Production questions:**
- What are the three most common agent failures in production and how do you prevent them? (Tool hallucination → schema validation; infinite loops → loop detection; context overflow → memory summarization)
- How do you evaluate an agent system? (Trajectory evaluation: task completion + efficiency + tool correctness + reasoning quality; run on standardized benchmarks like GAIA/SWE-bench)
- How would you reduce the cost of a complex agent that makes 50+ LLM calls per task? (Route cheap steps to smaller models; cache tool results; compress memory early; set budgets per step type)

**System design:**
- Design a coding agent that can fix GitHub issues end to end.
- Design a research agent that can produce a cited report from a natural language query.
- How would you handle prompt injection attacks in a web-browsing agent?

---

{: .highlight }
**Previous:** [ML Spine Ch B — LLM Internals →]({% link docs/ml-spine/chapter-llm.md %})
