from __future__ import annotations

from tinyrag.models import RetrievedChunk

INSUFFICIENT_CONTEXT_ANSWER = "Insufficient specification context to answer this question."


def format_context(results: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        lines.append(
            f"[{index}] product={chunk.product}; variant={chunk.variant}; field={chunk.field}; "
            f"value={chunk.value}; source={chunk.source_url}"
        )
    return "\n".join(lines)


def build_prompt(question: str, results: list[RetrievedChunk]) -> str:
    context = format_context(results)
    return f"""You are TinyRAG, a grounded product specification assistant.
/no_think
Answer only from the retrieved context.
Keep the answer concise: use one short sentence or compact bullets.
Preserve exact model names, numbers, symbols, units, and variants.
Do not repeat the question, repeat labels, or continue with new `Question:` / `Answer:` pairs.
Do not output hidden reasoning or `<think>` blocks.
Do not add explanations, comparisons, or capabilities that are absent from the retrieved context.
If the context does not contain the answer, reply exactly: {INSUFFICIENT_CONTEXT_ANSWER}
Use Traditional Chinese when the question is Traditional Chinese; otherwise answer in the user's language.

Retrieved context:
{context}

Question: {question}
Final answer:"""
