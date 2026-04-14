"""
Behavioral smoke test — requires Ollama running with qwen2.5:7b pulled.
Run with: pytest tests/behavioral/ -v -s
"""
import pytest
from pocket_mem import MemoryAgent, LLMConfig


@pytest.fixture(scope="module")
def warm_llm():
    """Fire a throwaway request so the model is loaded before timed tests run."""
    import requests
    try:
        requests.post(
            "http://localhost:11434/v1/chat/completions",
            json={"model": "qwen2.5:7b", "messages": [{"role": "user", "content": "hi"}]},
            timeout=120,
        )
    except Exception:
        pass


@pytest.fixture
def agent(tmp_path, warm_llm):
    llm = LLMConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b", timeout=120)
    a = MemoryAgent("smoke_test", path=str(tmp_path), llm=llm)
    yield a
    a.close()


def test_observe_and_recall_fact(agent):
    agent.observe(
        user_input="My boss is Sarah and she prefers Slack over email.",
        agent_response="Got it, I'll remember that about Sarah."
    )
    agent._executor.shutdown(wait=True)

    answer = agent.recall("Who is my boss?", mode="answer")
    print(f"\n[answer] {answer}")
    assert "sarah" in answer.lower(), f"Expected Sarah in answer, got: {answer}"


def test_recall_context_mode(agent):
    agent.observe(
        user_input="The project deadline is end of Q2.",
        agent_response="Noted."
    )
    agent._executor.shutdown(wait=True)

    ctx = agent.recall("When is the deadline?", mode="context")
    print(f"\n[context] {ctx}")
    assert "Q2" in ctx or "deadline" in ctx.lower(), f"Context missing deadline: {ctx}"


def test_topics_populated(agent):
    agent.observe(
        user_input="I use Python and prefer type hints everywhere.",
        agent_response="Understood."
    )
    agent._executor.shutdown(wait=True)

    topics = agent.topics()
    print(f"\n[topics] {topics}")
    assert len(topics) > 0, "Expected at least one topic after observe"


def test_stats_after_observe(agent):
    agent.observe(
        user_input="I live in Berlin and work remotely.",
        agent_response="Noted."
    )
    agent._executor.shutdown(wait=True)

    s = agent.stats()
    print(f"\n[stats] {s}")
    assert s["node_count"] > 0, f"Expected node_count > 0, got: {s}"
    assert s["edge_count"] >= 0
