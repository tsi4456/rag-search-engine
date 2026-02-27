import os
from time import sleep
from dotenv import load_dotenv
from google import genai
import json
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""

    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""

    response = client.models.generate_content(model=model, contents=prompt)
    rewritten = (response.text or "").strip().strip('"')
    return rewritten if rewritten else query


def expand_query(query: str) -> str:
    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""

    response = client.models.generate_content(model=model, contents=prompt)
    expanded_terms = (response.text or "").strip().strip('"')

    return f"{query} {expanded_terms}"


def enhance_query(query: str, method: str | None = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case "expand":
            return expand_query(query)
        case _:
            return query


def get_llm_response(text: str):
    resp = client.models.generate_content(model=model, contents=text)
    return resp.text


def rerank_results(query: str, results: list, method: str):
    match method:
        case "individual":
            return rerank_individual(query, results)
        case "batch":
            return rerank_batch(query, results)
        case "cross_encoder":
            return rerank_cross(query, results)
        case _:
            return results


def rerank_individual(query: str, results: list):
    for i, r in enumerate(results, 1):
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {r.get("title", "")} - {r.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Give me ONLY the number in your response, no other text or explanation.

        Score:"""
        resp = client.models.generate_content(model=model, contents=prompt)
        r["rerank"] = resp.text
        sleep(12.0)
    return sorted(results, key=lambda x: x["rerank"], reverse=True)


def rerank_batch(query: str, results: list):
    doc_list = [r["document"] for r in results]
    prompt = f"""Rank these movies by relevance to the search query.

        Query: "{query}"

        Movies:
        {doc_list}

        Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

        [75, 12, 34, 2, 1]
        """
    resp = client.models.generate_content(model=model, contents=prompt)
    ranks = {r: i for i, r in enumerate(json.loads(resp.text), 1)}
    for r in results:
        r["rerank"] = ranks[r["id"]]
    return sorted(results, key=lambda x: x["rerank"], reverse=True)


def rerank_cross(query: str, results: list):
    pairs = [
        [query, f"{r.get('title', '')} - {r.get('document', '')}"] for r in results
    ]
    cross = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross.predict(pairs)
    for r, s in zip(results, scores):
        r["rerank"] = s
    return sorted(results, key=lambda x: x["rerank"], reverse=True)


def evaluate_results(query: str, results: list):
    new_results = [f"{r['title']}: {r['document']}" for r in results]

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

        Query: "{query}"

        Results:
        {chr(10).join(new_results)}

        Scale:
        - 3: Highly relevant
        - 2: Relevant
        - 1: Marginally relevant
        - 0: Not relevant

        Do NOT give any numbers out than 0, 1, 2, or 3.

        Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

        [2, 0, 3, 2, 0, 1]"""

    resp = client.models.generate_content(model=model, contents=prompt)
    for r, s in zip(new_results, json.loads(resp.text)):
        r["eval"] = s
    return sorted(new_results, key=lambda x: x["eval"], reverse=True)
