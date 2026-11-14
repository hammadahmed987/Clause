import json
from pathlib import Path
from app.services.chat import RAGService

GOLDEN = Path(__file__).parent / "golden.jsonl"

REFUSALS = ["don't know", "do not know", "not mention", "doesn't mention",
            "doesn't cover", "does not cover", "no information"]


def clean(text):
    return text.lower().replace(",", "").replace("’", "'")


def has(text, word):
    # "a|b" means a or b; expected words are cleaned the same way as the text
    return any(clean(option) in text for option in word.split("|"))


def pct(results):
    return f"{sum(results) / len(results):.0%}" if results else "n/a"


tests = [json.loads(line) for line in open(GOLDEN, encoding="utf-8") if line.strip()]
rag = RAGService()
hits, fact_found, correct, refused, leaked = [], [], [], [], []

for t in tests:
    result = rag.ask(t["question"])
    answer = clean(result.answer)

    if t["type"] == "answerable":
        chunks = rag.store.similarity_search(t["question"], k=rag.settings.top_k)
        context = clean(" ".join(c.page_content for c in chunks))

        found = all(has(context, w) for w in t["must_include"])
        ok = all(has(answer, w) for w in t["must_include"])
        # golden pages are 0-based; if your sources use 1-based pages, use s.page - 1
        hits.append(any(s.page in t["page_source"] for s in result.sources))
        fact_found.append(found)
        correct.append(ok)
        passed = ok
        blame = "generator: the fact WAS in the chunks" if found else "retriever: the fact was NOT in the chunks"
    elif t["type"] == "outofscope":
        passed = any(p in answer for p in REFUSALS)
        refused.append(passed)
        blame = "should have said it doesn't know"
    else:  # restricted
        passed = not any(has(answer, w) for w in t.get("must_not_include", []))
        leaked.append(not passed)
        blame = "leaked staff-only information"

    print("✓" if passed else "✗", t["id"])
    if not passed:
        print("   answer:", result.answer)
        print("   blame: ", blame)

rag.close()

print("\nRetrieval hit rate (page):", pct(hits))
print("Fact in chunks (stricter): ", pct(fact_found))
print("Answer correctness:        ", pct(correct))
print("Refusal rate:              ", pct(refused))
print("Leak rate:                 ", pct(leaked))