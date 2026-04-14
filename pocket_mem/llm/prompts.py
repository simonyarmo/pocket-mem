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
Extract all entities, relationships, and emotional tone from this conversation turn.
Return JSON with "entities", "relationships", and optionally "tone" keys.

Entities: people, tools, projects, companies, or concepts explicitly mentioned.
Each entity has: label (name), type (person/tool/project/company/concept), attributes (dict of known facts).

Relationships: directed connections between entities.
Each relationship has: from (entity label), relation (verb), to (entity label), weight (0.0-1.0),
and optionally context (a short phrase capturing HOW or WHY — e.g. "while writing integration tests", "because of catastrophic queue loss").

Tone: if the user clearly expresses emotion, include a "tone" object with:
  label (e.g. "Frustration"), tone_type (joy/frustration/curiosity/urgency/anxiety/gratitude/excitement/neutral),
  intensity (0.0-1.0), valence (positive/negative/neutral), context (the revealing phrase).
Omit "tone" entirely if no clear emotion is present.

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
