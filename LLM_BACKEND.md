# LLM_BACKEND.md

## Config
```python
@dataclass
class LLMConfig:
    base_url:    str   = "http://localhost:11434/v1"
    model:       str   = "qwen2.5:7b"
    api_key:     str   = "ollama"
    timeout:     int   = 30
    temperature: float = 0.1   # low = consistent JSON
    max_tokens:  int   = 1024
```

## Model selection (RTX 3060 12GB)
All-or-nothing GPU rule: partial layer offloading is slower than pure CPU — don't split.

| Model | VRAM | Speed | Context | Use |
|-------|------|-------|---------|-----|
| `qwen2.5:7b` | ~5.5GB | 40–55 t/s | 32K | **Default** |
| `qwen2.5:14b` | ~10.5GB | 20–30 t/s | 4–6K | Better quality |
| `qwen2.5:14b` CPU | 0 | 4–6 t/s | 128K | Overnight evals |

```bash
curl -fsSL https://ollama.com/install.sh | sh && ollama pull qwen2.5:7b
ollama ps   # confirm 100% GPU, not split
```

## Structured output (equivalent to with_structured_output)

**1. JSON mode — use for classify + summarize:**
```python
{"model": ..., "messages": [...], "format": "json", "temperature": 0.1}
```

**2. JSON Schema — use for extract (fixed schema):**
```python
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "type":  {"type": "string", "enum": ["person","tool","project","company","concept"]},
                "attributes": {"type": "object"}
            }, "required": ["label","type"]
        }},
        "relationships": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "from": {"type": "string"}, "relation": {"type": "string"},
                "to":   {"type": "string"}, "weight": {"type": "number"}
            }, "required": ["from","relation","to"]
        }}
    }, "required": ["entities","relationships"]
}
# Pass schema as format value — not just "json"
{"model": ..., "messages": [...], "format": EXTRACTION_SCHEMA}
```

**3. Fallback (non-Ollama APIs):**
```python
try:    return json.loads(raw)
except: return json.loads(re.sub(r"```json|```", "", raw).strip())
```

## Switching providers
```python
# Claude
LLMConfig(base_url="https://api.anthropic.com/v1", model="claude-haiku-4-5-20251001", api_key=KEY)
# OpenAI
LLMConfig(base_url="https://api.openai.com/v1", model="gpt-4o-mini", api_key=KEY)
```

## Multi-model (v2 only)
Single model for v1. Ollama unloads/reloads on model switch — swap latency negates gains. If extraction quality is a problem, bump the single model to 14b first.
