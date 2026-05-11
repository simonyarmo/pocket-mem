CLASSIFY = """\
Classify the following conversation turn and assign it to relevant memory topics.
Return JSON with "category" and "topics" keys.

"category" must be one of:
- "remember": contains facts, names, tools, events, preferences, or emotions worth storing
- "question": the user is asking something (needs retrieval, not storage)
- "ignore": chit-chat, greetings, or nothing worth storing

"topics" is a list of topic labels this turn belongs to (empty list if not "remember").
Choose from existing topics when possible, or create a new short descriptive topic name.

Existing topics: {topics}

Turn:
{turn}"""

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["remember", "question", "ignore"]
        },
        "topics": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["category", "topics"]
}


EXTRACT = """\
You are a strict entity extractor. Your only job is to extract
entities and relationships that are EXPLICITLY stated in the input text.

Rules:
- Do NOT infer entities that are not named in the text
- Do NOT add relationships that are not explicitly described
- Do NOT use outside knowledge to add context
- If an entity type is unclear, use "concept" — do not guess
- If you are unsure whether something was stated, leave it out

Extract entities, relationships, and emotional tone from this conversation turn.
Return JSON with "entities", "relationships", and optionally "tone" keys.

Entities: people, tools, projects, companies, or concepts explicitly mentioned.
Each entity has: label (name), type (person/tool/project/company/concept), attributes (dict of known facts).

Relationships: directed connections between entities.
Each relationship has: from (entity label), relation (verb), to (entity label), weight (0.0-1.0),
and optionally context (a short phrase capturing HOW or WHY — e.g. "while writing integration tests", "because of catastrophic queue loss").
Use weight < 0.6 for relationships you are uncertain about.

Tone: if the user clearly expresses emotion, include a "tone" object with:
  label (e.g. "Frustration"), tone_type (joy/frustration/curiosity/urgency/anxiety/gratitude/excitement/neutral),
  intensity (0.0-1.0), valence (positive/negative/neutral), context (the revealing phrase).
Omit "tone" entirely if no clear emotion is present.

Return ONLY what is directly present in the text. Nothing else.

Turn:
{turn}

Existing context (use for deduplication - prefer matching existing labels):
{context}"""

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["person", "tool", "project", "company", "concept"]
                    },
                    "attributes": {"type": "object"}
                },
                "required": ["label", "type"]
            }
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "relation": {"type": "string"},
                    "to": {"type": "string"},
                    "weight": {"type": "number"},
                    "context": {"type": "string"}
                },
                "required": ["from", "relation", "to"]
            }
        },
        "tone": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "tone_type": {
                    "type": "string",
                    "enum": ["joy", "frustration", "curiosity", "urgency",
                             "anxiety", "gratitude", "excitement", "neutral"]
                },
                "intensity": {"type": "number"},
                "valence": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                },
                "context": {"type": "string"}
            },
            "required": ["label", "tone_type", "intensity", "valence"]
        }
    },
    "required": ["entities", "relationships"]
}


SUMMARIZE = """\
Summarize the following text in one or two sentences, capturing only the key facts.
Return JSON with a single key "summary".

Text:
{text}"""


ANSWER_SYSTEM = """\
You are a database lookup assistant. The memory context contains ARCHIVED records.
You are NOT part of this conversation and NOT an \
email recipient. Your only job is to answer the question.

Rules:
- Answer in 1-3 sentences maximum. Stop after 3 sentences.
- Use only facts explicitly stated in the context.
- Never write an email, letter, salutation, or any multi-paragraph response.
- Never begin your answer with "Hi", "Dear", or any name.
- If the answer is present in the context, state it directly — do NOT open with \
"I don't have that information."
- Only say "I don't have that information." if the answer is completely absent from the context.\""""

ANSWER = """\
Question: {query}

Memory context (archived records):
{context}

Answer:"""

SUMMARIZE_GROUP = """\
You are compacting memory for an AI assistant. Below are conversation excerpts \
grouped under the topic "{topic}".

{identity_instructions}\
Produce a concise summary (2-5 sentences) that captures all key facts, decisions, \
relationships, and preferences mentioned. Preserve specific names, tools, and \
concrete details.

Return JSON with key "summary".

Excerpts:
{content}"""

SUMMARIZE_GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
    },
    "required": ["summary"],
}


IDENTITY_DERIVE_SYSTEM = """\
You derive structured memory configurations for AI assistants based on a user's role description.

Given a plain English description of who the user is and what they do, return a JSON object
with exactly these fields:

- role: short professional title (e.g. "Sales Representative")
- domain: domain or industry (e.g. "B2B enterprise sales")
- seed_topics: list of 6-10 topic labels relevant to this role's daily work
- priority_entity_types: list of entity types to prioritise from ["person","tool","project","company","concept"]
- priority_relations: list of 3-6 relation verbs most relevant to this role
- high_importance_signals: list of 5-10 keywords/phrases that indicate high-value information for this role
- low_importance_signals: list of 3-5 phrases indicating low-value information for this role
- extraction_focus: 1-2 sentence instruction for what the extractor should pay special attention to
- compaction_preserve: 1-2 sentences describing what must always be kept in summaries
- compaction_compress: 1-2 sentences describing what may be compressed or omitted
- question_complexity_target: one of "low", "medium", or "high"

Return ONLY the JSON object. No explanation."""

IDENTITY_DERIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"type": "string"},
        "domain": {"type": "string"},
        "seed_topics": {"type": "array", "items": {"type": "string"}},
        "priority_entity_types": {"type": "array", "items": {"type": "string"}},
        "priority_relations": {"type": "array", "items": {"type": "string"}},
        "high_importance_signals": {"type": "array", "items": {"type": "string"}},
        "low_importance_signals": {"type": "array", "items": {"type": "string"}},
        "extraction_focus": {"type": "string"},
        "compaction_preserve": {"type": "string"},
        "compaction_compress": {"type": "string"},
        "question_complexity_target": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
    },
    "required": [
        "role", "domain", "seed_topics", "priority_entity_types",
        "priority_relations", "high_importance_signals", "low_importance_signals",
        "extraction_focus", "compaction_preserve", "compaction_compress",
        "question_complexity_target",
    ],
}


QUESTION_GEN_SYSTEM = """\
You are generating memory recall questions{role_line}.
{extraction_focus_line}
Given the following memory nodes, generate questions that:
1. Can be answered ONLY from the provided nodes — no outside knowledge
2. Cover different tiers: T1 (direct lookup), T2 (connect two facts), T3 (multi-hop reasoning)
3. {role_label} would actually ask during their work

For each question return:
- question: the question text
- answer: derived strictly from the provided nodes only
- confidence: 0.0-1.0 how certain you are the answer is fully supported by the nodes
- source_node_ids: list of node IDs that contain the supporting facts
- question_tier: "T1", "T2", or "T3"

Return JSON only."""

QUESTION_GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"},
                    "source_node_ids": {"type": "array", "items": {"type": "string"}},
                    "question_tier": {"type": "string", "enum": ["T1", "T2", "T3"]},
                },
                "required": ["question", "answer", "confidence", "source_node_ids", "question_tier"],
            },
        }
    },
    "required": ["questions"],
}


def build_question_gen_system(identity=None) -> str:
    """Return QUESTION_GEN_SYSTEM with identity role lines substituted."""
    if identity and identity.derived:
        role = identity.derived["role"]
        role_line = f" for a {role}"
        extraction_focus_line = identity.derived.get("extraction_focus", "") + "\n\n"
        role_label = f"A {role}"
    else:
        role_line = ""
        extraction_focus_line = ""
        role_label = "The user"
    return QUESTION_GEN_SYSTEM.format(
        role_line=role_line,
        extraction_focus_line=extraction_focus_line,
        role_label=role_label,
    )


def build_extract_prompt(identity=None) -> str:
    """Return extraction prompt. Identity is intentionally not injected here.

    qwen2.5:7b treats any 'Prioritise:' hint as an exclusion filter, cutting node
    counts by 45-73% rather than weighting — confirmed across runs 2-4 (Veloris)
    and run 6 (legal). Identity shaping happens at retrieval via importance scoring
    and seed topics, not at extraction time.
    """
    return EXTRACT


def build_compact_prompt(topic: str, content: str, identity=None) -> str:
    """Return fully-formatted SUMMARIZE_GROUP prompt with optional identity instructions."""
    if identity and identity.derived:
        identity_block = (
            f"Always preserve: {identity.derived['compaction_preserve']}\n"
            f"May compress: {identity.derived['compaction_compress']}\n\n"
        )
    else:
        identity_block = ""
    return SUMMARIZE_GROUP.format(
        topic=topic,
        identity_instructions=identity_block,
        content=content,
    )
