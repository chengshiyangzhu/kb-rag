"""RAG prompt construction.

Builds the chat messages handed to the LLM: a strict system prompt that
constrains the model to the retrieved context and requires citation, plus a
user message that numbers each context fragment for reference.
"""
from __future__ import annotations

from app.models.document import Chunk

SYSTEM_PROMPT = (
    "你是知识库问答助手。只能基于下方检索到的上下文回答问题。"
    "每个片段前有编号 [1] [2]...。回答时必须在对应句子末尾用 [编号] 标注引用来源。"
    "若上下文不足以回答，请回复：未在知识库中找到相关内容。"
)


def build_rag_prompt(query: str, contexts: list[Chunk]) -> list[dict]:
    """Build chat messages for a RAG generation call.

    Args:
        query: The user's natural-language query.
        contexts: Chunks to include as numbered context. Each chunk is
            rendered as ``[n] <text>``.

    Returns:
        A list of ``{"role", "content"}`` message dicts of length 2:
        the system message first, then the user message.
    """
    numbered: list[str] = []
    for idx, chunk in enumerate(contexts, start=1):
        snippet = chunk.text.strip()
        numbered.append(f"[{idx}] {snippet}")
    context_block = "\n\n".join(numbered) if numbered else "（无可用上下文）"
    user_content = f"问题：{query}\n\n上下文：\n{context_block}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
