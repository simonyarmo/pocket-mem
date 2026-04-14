# pocket-mem

**Persistent knowledge graph memory for AI agents — drop in with two lines, runs in the background, no external database required.**

pocket-mem gives any AI agent a long-term memory that works like a knowledge graph. When your agent has a conversation, pocket-mem silently extracts the people, tools, decisions, and relationships mentioned and stores them in a structured local database. The next time your agent needs context — even sessions later — it can recall exactly who David is, what tools he recommended, and what you decided about the database last Tuesday.

It works with any agent, any LLM, and any Python project. There is no server to run, no cloud account to create, and no API key required beyond your LLM of choice.

---

## Table of contents
- [How it works](#how-it-works)
- [System requirements](#system-requirements)
- [Installation](#installation)
- [Setting up Ollama](#setting-up-ollama)
- [Choosing a model](#choosing-a-model)
- [Quick start](#quick-start)
- [Wiring memory into your agent](#wiring-memory-into-your-agent)
- [Recall modes](#recall-modes)
- [Visualizing memory](#visualizing-memory)
- [Sharing memory](#sharing-memory)
- [Storage options](#storage-options)

---

## How it works

Every time your agent receives a message and produces a response, you call `agent.observe()`. This runs in a background thread and never blocks your agent. Under the hood it:

1. Classifies the content into topics (like "People I Know", "Dev Tools", "Decisions")
2. Extracts named entities — people, tools, projects — and the typed relationships between them
3. Stores everything in a local SQLite knowledge graph with vector embeddings for semantic search

When your agent needs memory, you call `agent.recall()`. This runs a hybrid search — keyword matching plus semantic vector similarity — and returns results in whichever format you need.

The result is an agent that remembers across sessions without you managing any of it.

---

## System requirements

### Minimum (CPU only)
- Python 3.10+
- 8 GB RAM
- ~500 MB disk for the embedding model
- Any modern CPU (4+ cores recommended)

### Recommended (GPU inference)
- Python 3.10+
- 16 GB RAM
- NVIDIA GPU with **at least 6 GB VRAM**
- NVIDIA drivers 525+ and CUDA 12.1+

### Optimal
- 32 GB RAM
- NVIDIA GPU with **12 GB VRAM** (RTX 3060, RTX 3070 Ti, RTX 4060 Ti 16GB, or better)

> **Important — all-or-nothing GPU rule:** Ollama either loads the entire model onto your GPU or falls back to CPU. Partial offloading (splitting layers between GPU and CPU) is actually *slower* than pure CPU because of PCIe transfer overhead. If the model doesn't fit in your VRAM, see [Choosing a model](#choosing-a-model) to pick a smaller model that does.

---

## Installation

```bash
pip install pocket-mem
```

That installs the package and the `all-MiniLM-L6-v2` embedding model (~22 MB, runs locally on CPU). The only additional setup is an LLM — see the next section.

---

## Setting up Ollama

Ollama is the recommended way to run a local LLM. It's free, runs entirely on your machine, and pocket-mem connects to it automatically with no configuration.

### Step 1 — Install Ollama

**Linux or WSL2:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```
Or download the app from [ollama.com/download](https://ollama.com/download).

**Windows:**
Download and run the installer from [ollama.com/download](https://ollama.com/download).

### Step 2 — Start Ollama

```bash
ollama serve
```

Leave this running in a terminal. Ollama listens on `http://localhost:11434`. On Linux you can run it as a background service instead:

```bash
sudo systemctl enable ollama
sudo systemctl start ollama
```

### Step 3 — Pull the default model

```bash
ollama pull qwen2.5:7b
```

This downloads the model (~4.7 GB compressed). You only need to do this once.

### Step 4 — Verify everything works

```bash
ollama list
# Should show qwen2.5:7b in the list

ollama run qwen2.5:7b "Respond with valid JSON: {\"status\": \"ok\"}"
# Should return a JSON response
```

If you get a JSON response, Ollama is set up correctly and pocket-mem will work.

---

## Choosing a model

### Default: `qwen2.5:7b` (recommended)

The default model is `qwen2.5:7b`. It excels at structured JSON output — the most critical capability for accurately extracting entities and relationships from text. If your hardware can run it, use it.

```bash
ollama pull qwen2.5:7b
```

This runs fully on GPU with 6 GB+ VRAM, or falls back to CPU at 4–6 t/s. On Apple Silicon, your full system RAM is available so 16 GB+ M-series Macs are ideal.

### Smaller model: `qwen2.5:3b`

If `qwen2.5:7b` doesn't fit in your VRAM or runs too slowly, you can step down:

```bash
ollama pull qwen2.5:3b
```

> **Warning:** The 3B model will extract less detail from conversations and is more prone to hallucination during ingestion. Entity extraction and relationship mapping will be noticeably less accurate. Use it only if you cannot run the 7B model.

```python
from pocket_mem import MemoryAgent, LLMConfig

agent = MemoryAgent(
    project="my-app",
    llm=LLMConfig(model="qwen2.5:3b")
)
```

### Cloud models (OpenAI, Claude, any OpenAI-compatible provider)

You can use any cloud LLM for both ingestion and recall. pocket-mem uses the OpenAI-compatible chat completions API, so it works with any provider that exposes it.

Cloud models give you the highest extraction quality with no local GPU requirement.

```python
import os
from pocket_mem import MemoryAgent, LLMConfig

# Anthropic Claude Haiku — excellent JSON extraction, low cost
agent = MemoryAgent(
    project="my-app",
    llm=LLMConfig(
        base_url="https://api.anthropic.com/v1",
        model="claude-haiku-4-5-20251001",
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )
)

# OpenAI GPT-4o Mini
agent = MemoryAgent(
    project="my-app",
    llm=LLMConfig(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"]
    )
)

# Any OpenAI-compatible provider (Groq, Together AI, Mistral, etc.)
agent = MemoryAgent(
    project="my-app",
    llm=LLMConfig(
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"]
    )
)
```

---

## Quick start

```python
from pocket_mem import MemoryAgent

# Creates a ./memory/ folder in your project directory automatically
agent = MemoryAgent(project="my-app")

# Store a conversation turn — non-blocking, returns immediately
agent.observe(
    user_input="My boss David recommended I try Cursor IDE for coding",
    agent_response="Got it, I'll keep that in mind."
)

# Recall memory as context — best method for injecting into system prompts
context = agent.recall("What tools has David recommended?", mode="context")
print(context)
# → "Entity: David (boss) — recommended Cursor IDE for coding."

# See what topics are stored in memory
print(agent.topics())
# → ["People I Know", "AI Tools"]

# Inspect the raw graph (useful for debugging)
import json
print(json.dumps(agent.recall("David", mode="raw"), indent=2))
```

---

## Wiring memory into your agent

There are two patterns for connecting pocket-mem to your existing LLM or agent. Both use the same API.

### Pattern A — Proactive injection (recommended for most agents)

Call `recall()` before every LLM call and inject the results into your system prompt. Your agent always has relevant memory in context without needing to explicitly ask for it.

```python
from pocket_mem import MemoryAgent

memory = MemoryAgent(project="my-app")

def chat(user_message: str) -> str:
    # 1. Retrieve relevant memory for this message
    context = memory.recall(user_message, mode="context")

    # 2. Inject into your system prompt
    system_prompt = f"""You are a helpful coding assistant.

## What you remember from past conversations
{context}

Use this context to give personalized, informed responses.
"""

    # 3. Call your LLM as normal
    response = your_llm.chat(system=system_prompt, user=user_message)

    # 4. Store this turn (non-blocking)
    memory.observe(user_input=user_message, agent_response=response)

    return response
```

This works with any LLM — OpenAI, Anthropic, local models, LangChain, anything that accepts a system prompt.

### Pattern B — Tool call (for autonomous or multi-step agents)

Expose `recall` to your LLM as a callable tool. The model decides when memory is relevant and calls it on demand.

```python
from pocket_mem import MemoryAgent

memory = MemoryAgent(project="my-app")

# Get an OpenAI-compatible tool definition — works with any compatible API
tools = [memory.as_tool()]

# Handle the tool call in your agent loop
def handle_tool_call(tool_name: str, args: dict) -> str:
    if tool_name == "recall_memory":
        return memory.recall(
            query=args["query"],
            mode=args.get("mode", "context")
        )
```

### Which pattern to use

| Agent type | Use |
|------------|-----|
| Conversational assistant | Pattern A |
| Coding assistant | Pattern A |
| Autonomous agent | Pattern B |
| Research agent | Pattern B |
| Not sure | Pattern A — simpler, works for most cases |

---

## Recall modes

The `mode` parameter controls what `recall()` returns.

### `mode="context"` — recommended

Returns a formatted string of relevant memories ready to inject directly into a system prompt. No LLM call — purely graph retrieval and formatting. **This is the best way to use pocket-mem.** It's fast, deterministic, and works with any downstream LLM you're already using.

```python
context = memory.recall("What database did we decide on?", mode="context")
# → "Decision (Jan 14): Chose PostgreSQL over SQLite — needs concurrent writes."

# Inject directly into your system prompt:
system = f"You are a helpful assistant.\n\n## Memory\n{context}"
```

### `mode="answer"`

Makes an LLM call to synthesize a natural language answer directly from memory. Best used when the user is asking a memory-specific question and you want pocket-mem to answer it directly rather than injecting context into another model.

```python
answer = memory.recall("Who recommended httpx?", mode="answer")
# → "David, your boss, mentioned httpx is better than requests for async HTTP work."
```

> **For best results, use Claude Haiku.** In benchmarks against the Veloris dataset (50 scored questions across direct lookup, single-hop, and multi-hop categories, plus 10 unanswerable), using `qwen2.5:7b` for ingestion and Claude Haiku for synthesis achieves a **2-run average of 98% accuracy** on answerable questions (peak 99%), with zero false positives on unanswerable ones. See [`tests/simulation/first_sim_test_50_q/BENCHMARK.md`](tests/simulation/first_sim_test_50_q/BENCHMARK.md) for the full results.
>
> Local models like `qwen2.5:7b` can answer memory questions but are more prone to synthesizing plausible-sounding answers that aren't supported by the stored facts.

If you want to keep `qwen2.5:7b` for ingestion (fast, free, local) but use Claude Haiku only when `mode="answer"` is called, set the `answer_*` fields separately:

```python
import os
from pocket_mem import MemoryAgent, LLMConfig

agent = MemoryAgent(
    project="my-app",
    llm=LLMConfig(
        # Ingestion — local Ollama, used for observe() and recall(mode="context")
        base_url="http://localhost:11434/v1",
        model="qwen2.5:7b",

        # Answer model — only used when recall(mode="answer") is called
        answer_base_url="https://api.anthropic.com/v1",
        answer_model="claude-haiku-4-5-20251001",
        answer_api_key=os.environ["ANTHROPIC_API_KEY"],
    )
)

answer = agent.recall("What did we decide about the auth system?", mode="answer")
```

If you omit the `answer_*` fields, `mode="answer"` uses the same model as ingestion. If you set `base_url` and `model` directly to a cloud model, that model is used for everything including extraction.

### `mode="raw"`

Returns the raw graph data as a Python list of nodes and edges. No LLM call, no formatting. Use this to debug what's actually stored.

```python
import json
data = memory.recall("David", mode="raw")
print(json.dumps(data, indent=2))
```

---

## Visualizing memory

pocket-mem includes a built-in graph explorer that opens in your browser. It shows every node and edge in your memory graph, with filtering by topic, node type, date, and keyword search.

```bash
# Open the visualizer for the default memory path
pocket-mem show

# Specify a project
pocket-mem show --project my-app

# Filter to a specific topic
pocket-mem show --project my-app --topic "People I Know"

# Filter by node type
pocket-mem show --project my-app --type entity

# Show only nodes updated in the last 7 days
pocket-mem show --project my-app --since 7d

# Pre-fill the search bar
pocket-mem show --project my-app --search David
```

The visualizer is read-only and requires no additional dependencies beyond the base install.

---

## Sharing memory

Memory is stored as a portable file. You can share your agent's full context with someone else.

### Export and import

Package your memory into a single `.mempack` file and send it to a colleague. Their agent picks up exactly where yours left off — all the people, decisions, tools, and relationships your agent has learned.

```python
# Export your memory
agent = MemoryAgent(project="my-project")
agent.export("project_memory.mempack")

# Your colleague imports it on their machine
their_agent = MemoryAgent(project="my-project")
their_agent.import_pack("project_memory.mempack")

# Their agent now has all your memory
print(their_agent.recall("What database did we decide on?", mode="answer"))
```

A `.mempack` file is a zip archive containing the SQLite database. It's self-contained and portable — you can email it, commit it to version control as a checkpoint, or back it up like any other file.

---

## Storage options

By default pocket-mem creates a `memory/` folder in your current directory and stores the database there. No configuration needed.

```
your-project/
├── main.py
├── memory/              ← created automatically
│   └── my-app.db
└── ...
```

**Change the storage location:**
```python
# Different local directory
agent = MemoryAgent(project="my-app", path="./data/memory/")

# Absolute path
agent = MemoryAgent(project="my-app", path="/home/user/shared-memory/")
```

Cloud storage (shared multi-user memory graphs) is planned for v2.

---

## License

MIT
