"""
Debug script — prints cosine similarity scores for all prebuilt configs
against a given description, so you can see exactly why a match passes or fails.

Run from repo root:
    python tests/simulation/debug_identity_match.py
"""
from pocket_mem.embedding import cosine_similarity, embed
from pocket_mem.identities.prebuilt import _load_prebuilts, _MATCH_THRESHOLD, _MIN_MARGIN

DESCRIPTIONS = [
    (
        "paralegal",
        "Paralegal at Castellan & Briggs LLP, a litigation law firm in Chicago. "
        "My job is to read all case notes every day -- tracking clients, opposing "
        "parties, attorneys, deadlines, damages, filings, and anything happening "
        "across active cases so I can brief the attorneys and answer questions "
        "about case status, key facts, and upcoming deadlines."
    ),
    (
        "executive_assistant",
        "Executive assistant at Veloris Technologies, a B2B SaaS company. "
        "My job is to stay informed by reading all company emails every day -- "
        "tracking people, projects, decisions, costs, and anything happening "
        "across the engineering and product teams so I can brief leadership "
        "and answer questions about what is going on."
    ),
    (
        "vectra",
        "Personal AI assistant named Vectra managing Jordan's daily life in Austin, "
        "Texas. I listen to all voice interactions, emails, and text messages to track "
        "Jordan's schedule, reminders, appointments, shopping lists, financial tasks, "
        "contacts, and anything that needs to be remembered or acted on."
    ),
]

prebuilts = _load_prebuilts()
print(f"Loaded {len(prebuilts)} prebuilt configs\n")

for label, desc in DESCRIPTIONS:
    print(f"=== {label} ===")
    print(f"Description: {desc[:80]}...")
    desc_emb = embed(desc)
    scores = []
    for cfg in prebuilts:
        role_emb = embed(cfg["role"])
        score = cosine_similarity(desc_emb, role_emb)
        scores.append((score, cfg["role"]))
    scores.sort(reverse=True)

    best_score, best_role = scores[0]
    second_score, second_role = scores[1]
    margin = best_score - second_score
    passes = best_score >= _MATCH_THRESHOLD and margin >= _MIN_MARGIN

    print(f"\n{'Rank':<5} {'Score':<8} Role")
    print("-" * 40)
    for i, (score, role) in enumerate(scores, 1):
        flag = " <- BEST" if i == 1 else (" <- 2nd" if i == 2 else "")
        print(f"{i:<5} {score:.4f}   {role}{flag}")

    print(f"\nThreshold : {_MATCH_THRESHOLD}  (score must be >= this)")
    print(f"Min margin: {_MIN_MARGIN}  (best must beat 2nd by >= this)")
    print(f"Best score: {best_score:.4f}  ({best_role})")
    print(f"2nd score : {second_score:.4f}  ({second_role})")
    print(f"Margin    : {margin:.4f}")
    print(f"Result    : {'MATCH -> ' + best_role if passes else 'NO MATCH'}")
    print()
