from memory_agent.llm.prompts import CLASSIFY, EXTRACT, SUMMARIZE, ANSWER


def test_classify_is_string():
    assert isinstance(CLASSIFY, str)
    assert len(CLASSIFY) > 20


def test_classify_has_turn_placeholder():
    assert "{turn}" in CLASSIFY


def test_extract_is_string():
    assert isinstance(EXTRACT, str)
    assert len(EXTRACT) > 20


def test_extract_has_required_placeholders():
    assert "{turn}" in EXTRACT
    assert "{context}" in EXTRACT


def test_summarize_is_string():
    assert isinstance(SUMMARIZE, str)
    assert len(SUMMARIZE) > 20


def test_summarize_has_text_placeholder():
    assert "{text}" in SUMMARIZE


def test_answer_is_string():
    assert isinstance(ANSWER, str)
    assert len(ANSWER) > 20


def test_answer_has_required_placeholders():
    assert "{query}" in ANSWER
    assert "{context}" in ANSWER


def test_prompts_are_distinct():
    prompts = [CLASSIFY, EXTRACT, SUMMARIZE, ANSWER]
    assert len(set(prompts)) == 4  # all unique


from memory_agent.llm.prompts import (
    CLASSIFY, EXTRACT, SUMMARIZE, ANSWER,
    CLASSIFY_SCHEMA, EXTRACTION_SCHEMA,
)


def test_classify_has_topics_placeholder():
    assert "{topics}" in CLASSIFY


def test_classify_schema_is_dict():
    assert isinstance(CLASSIFY_SCHEMA, dict)
    assert "category" in CLASSIFY_SCHEMA["properties"]
    assert "topics" in CLASSIFY_SCHEMA["properties"]
    assert CLASSIFY_SCHEMA["required"] == ["category", "topics"]


def test_extraction_schema_is_dict():
    assert isinstance(EXTRACTION_SCHEMA, dict)
    assert "entities" in EXTRACTION_SCHEMA["properties"]
    assert "relationships" in EXTRACTION_SCHEMA["properties"]
    assert "entities" in EXTRACTION_SCHEMA["required"]
    assert "relationships" in EXTRACTION_SCHEMA["required"]
    assert "tone" not in EXTRACTION_SCHEMA.get("required", [])


def test_extract_mentions_tone():
    assert "tone" in EXTRACT.lower()
