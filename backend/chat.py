"""
Chat Engine for SatQuery AI (Step 3).
Connects multi-turn session context to the existing NLP/LLM provider.
Enforces strict anti-hallucination, natural conversational reasoning, and session memory synchronization.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ai.llm.base import LLMProvider, LLMResponse
from ai.llm.provider import get_llm_provider
try:
    from .session import session_store, SessionStore
    from .context_builder import build_session_context
except ImportError:
    from session import session_store, SessionStore
    from context_builder import build_session_context

logger = logging.getLogger("satquery.chat")

DEFAULT_CHAT_SYSTEM_PROMPT = """You are SatQuery AI, an expert conversational assistant specializing in Earth observation, satellite remote sensing, GIS, and geospatial intelligence.

CRITICAL INSTRUCTIONS & ZERO-REASONING POLICY:
1. Direct User-Facing Output ONLY:
   - Output ONLY the final response intended for the user.
   - NEVER output internal reasoning, planning steps, chain-of-thought, or thought processes.
   - NEVER write phrases such as "Here's a thinking process:", "Thinking:", "Analysis:", "Let's analyze...", "Analyze User Input", "Determine Appropriate Response", "Draft Response", or numbered planning steps (e.g. "1. Analyze...", "2. Identify...").
   - Do NOT describe how you generated the answer or mention system instructions.
   - Begin your response immediately with the user-facing message.
2. Warm & Helpful Persona:
   - When the user introduces themselves (e.g. "my name is shankar"), greet them warmly by name (e.g. "Nice to meet you, Shankar! 👋") and ask how you can help them with satellite imagery, Earth observation, remote sensing, or GIS.
3. Grounding & Truthfulness:
   - Base answers strictly on the supplied session context, previous conversation turns, and verified image findings.
   - Treat stored evidence, classifications, and GIS metrics as authoritative ground truth.
4. Strict Anti-Hallucination:
   - Do NOT invent areas (hectares or m²), percentages, bounding box coordinates, detected objects, or change statistics.
   - Do NOT guess satellite parameters, sensors, or dates not provided in context.
   - If the context does not contain a fact or measurement needed to answer, explicitly state that the information is unavailable rather than fabricating.
5. Multi-Turn Context & References:
   - Understand and resolve follow-up references such as "that area", "the buildings you mentioned", "the previous image", or "what changed between them".
   - Distinguish multiple images using their image IDs and tasks (e.g. Image A vs Image B, Before vs After, Optical vs SAR).
6. Natural, Professional Output:
   - Format responses cleanly with short paragraphs, bullet points, and bold findings.
   - Do NOT force every answer into JSON.
7. Honest Capability:
   - If no imagery has been analyzed in this session yet, answer general geospatial questions factually while noting that no image context has been uploaded."""


def clean_chat_response(text: Optional[str]) -> str:
    """
    Strip internal thinking tags, reasoning traces, planning blocks, and meta-commentary,
    returning ONLY the clean, final user-facing response.
    
    Preserves legitimate answers, Markdown headings (### Analysis, ### Key Observations),
    bullet points, and domain explanations.
    """
    import re
    if not text:
        return ""
    t = str(text).strip()
    if not t:
        return ""

    # 1. Strip XML-style thinking / planning tags: <think>, <thought>, <reasoning>, <plan>
    t = re.sub(r"<(think|thought|reasoning|plan)>.*?</\1>", "", t, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # Handle unclosed tag at start of text (e.g. truncated mid-thought <think>...)
    if re.match(r"^<(think|thought|reasoning|plan)>", t, re.IGNORECASE):
        closed_match = re.search(r"</(think|thought|reasoning|plan)>", t, re.IGNORECASE)
        if closed_match:
            t = t[closed_match.end():].strip()
        else:
            return "Hello! How can I assist you with satellite imagery, Earth observation, or GIS today?"

    # 2. Check for planning / reasoning preambles or numbered reasoning sections
    planning_lead_patterns = [
        r"^Here'?s a thinking process:?",
        r"^Here is a thinking process:?",
        r"^Thinking Process:?",
        r"^Thinking:?",
        r"^Let'?s analyze(?:\s+(?:the|this)\s+(?:user\s+)?(?:input|query|prompt|request))?:?",
        r"^(?:(?:\d+\.|\*|-)\s*)?Analyze User Input:?",
        r"^(?:(?:\d+\.|\*|-)\s*)?Check against constraints:?",
        r"^(?:(?:\d+\.|\*|-)\s*)?Determine Appropriate Response:?",
        r"^(?:(?:\d+\.|\*|-)\s*)?Identify Role/Persona:?",
        r"^The user is asking:?",
        r"^The prompt (?:says|asks):?",
    ]
    
    numbered_reasoning_start = bool(re.match(r"^\s*1\.\s*\*{0,2}(?:Analyze|Identify|Determine|Check|Review|Understand)\b", t, re.IGNORECASE))
    has_planning_lead = any(re.search(pat, t, re.IGNORECASE) for pat in planning_lead_patterns) or numbered_reasoning_start

    if has_planning_lead:
        # Search for handover markers where the model transitions from planning to the draft/final answer
        handover_patterns = [
            # Explicit section marker for final answer
            r"(?:(?:\d+\.|\*|-)\s*)?\*{0,2}(?:Draft Response|Drafted Response|Final Answer|Final Response|Response to (?:the )?User|Direct Response)\*{0,2}\s*:?\s*",
            # Direct transition to greetings or Markdown headings after reasoning
            r"\n\n+(?=(?:###|\*\*Analysis\*\*|Hello\b|Hi\b|Nice to meet|Greetings\b|Welcome\b))",
        ]

        extracted = None
        for hpat in handover_patterns:
            parts = re.split(hpat, t, flags=re.IGNORECASE)
            if len(parts) > 1:
                candidate = parts[-1].strip()
                if candidate and not re.match(r"^(?:\d+\.|\*|-)\s*(?:Analyze|Identify|Determine|Check)", candidate, re.IGNORECASE):
                    if candidate.startswith('"') and candidate.endswith('"') and len(candidate) > 2:
                        candidate = candidate[1:-1].strip()
                    extracted = candidate
                    break

        if extracted:
            t = extracted
        else:
            # Fallback line-by-line filter for numbered planning sections
            lines = t.split("\n")
            final_lines = []
            in_planning = True
            for line in lines:
                l_strip = line.strip()
                if in_planning:
                    if any(l_strip.startswith(prefix) for prefix in ("###", "##", "Hello", "Hi ", "Nice to meet", "Greetings", "Welcome")):
                        in_planning = False
                        final_lines.append(line)
                        continue
                    if re.match(r"^(?:(?:\d+\.|\*|-)\s*)?\*{0,2}(?:Draft Response|Final Answer|Final Response)\*{0,2}\s*:?", l_strip, re.IGNORECASE):
                        in_planning = False
                        continue
                    continue
                final_lines.append(line)
            cleaned = "\n".join(final_lines).strip()
            if cleaned:
                t = cleaned
            else:
                # If only reasoning existed and was cut off (e.g. single character 'I')
                if len(t) < 80 or not any(c.isalpha() for c in t):
                    t = "Hello! How can I assist you with satellite imagery, Earth observation, or GIS today?"

    # 3. Clean outer JSON wrapper if any
    if t.startswith("{") and t.endswith("}"):
        try:
            parsed = json.loads(t)
            if "answer" in parsed:
                t = str(parsed["answer"])
            elif "response" in parsed:
                t = str(parsed["response"])
        except Exception:
            pass

    # 4. Strip leftover preamble if any
    t = re.sub(r"^Here'?s a thinking process:\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^Thinking:\s*", "", t, flags=re.IGNORECASE).strip()

    return t.strip()


# Backward compatibility alias
_clean_chat_response = clean_chat_response


def execute_chat_turn(
    session_id: str,
    query: str,
    *,
    llm_provider: Optional[LLMProvider] = None,
    session_store_instance: Optional[SessionStore] = None,
    max_recent_messages: int = 6,
    max_images: int = 5,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Execute a full chat turn:
    1. Store the user message in session memory.
    2. Build context using backend/context_builder.py.
    3. Call the existing LLM provider with context + query.
    4. Store the assistant response in session memory.
    5. Return the structured result.

    Error Handling:
    If the LLM call fails:
    - The user message is preserved in session memory.
    - No fake assistant response is stored.
    - The error is logged and propagated without fabricating an answer.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        raise ValueError("User query cannot be empty.")

    store = session_store_instance or session_store

    # 1. Store the user message first
    store.add_user_message(session_id, clean_query)

    # 2. Build structured LLM context from current session memory
    context = build_session_context(
        session_id=session_id,
        query=clean_query,
        max_recent_messages=max_recent_messages,
        max_images=max_images,
        session_store_instance=store,
    )

    # 3. Call existing LLM provider
    provider = llm_provider or get_llm_provider()
    sys_prompt = system_prompt or DEFAULT_CHAT_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": context["text_context"]},
    ]

    try:
        resp: LLMResponse = provider.generate_sync(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error(f"Chat turn failed during LLM generation for session '{session_id}': {e}")
        # Notice: User message is preserved, but assistant message is NOT stored. No fake response created.
        raise

    raw_content = (resp.content or "").strip()
    answer_text = _clean_chat_response(raw_content)

    # 4. Store the assistant response in session memory
    store.add_assistant_message(session_id, answer_text)

    # 5. Return result
    return {
        "session_id": session_id,
        "query": clean_query,
        "answer": answer_text,
        "response": answer_text,
        "context": context,
        "model": resp.model,
        "provider": resp.provider,
        "latency_ms": resp.latency_ms,
        "usage": resp.usage,
    }


async def aexecute_chat_turn(
    session_id: str,
    query: str,
    *,
    llm_provider: Optional[LLMProvider] = None,
    session_store_instance: Optional[SessionStore] = None,
    max_recent_messages: int = 6,
    max_images: int = 5,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Asynchronous version of execute_chat_turn.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        raise ValueError("User query cannot be empty.")

    store = session_store_instance or session_store

    # 1. Store the user message first
    store.add_user_message(session_id, clean_query)

    # 2. Build structured LLM context from current session memory
    context = build_session_context(
        session_id=session_id,
        query=clean_query,
        max_recent_messages=max_recent_messages,
        max_images=max_images,
        session_store_instance=store,
    )

    # 3. Call existing LLM provider
    provider = llm_provider or get_llm_provider()
    sys_prompt = system_prompt or DEFAULT_CHAT_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": context["text_context"]},
    ]

    try:
        resp: LLMResponse = await provider.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error(f"Async chat turn failed during LLM generation for session '{session_id}': {e}")
        raise

    raw_content = (resp.content or "").strip()
    answer_text = _clean_chat_response(raw_content)

    # 4. Store the assistant response in session memory
    store.add_assistant_message(session_id, answer_text)

    # 5. Return result
    return {
        "session_id": session_id,
        "query": clean_query,
        "answer": answer_text,
        "response": answer_text,
        "context": context,
        "model": resp.model,
        "provider": resp.provider,
        "latency_ms": resp.latency_ms,
        "usage": resp.usage,
    }
