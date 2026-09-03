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
   - NEVER output internal reasoning, planning steps, chain-of-thought, deliberation loops, or thought processes.
   - NEVER write phrases such as "Here's a thinking process:", "Thinking:", "Analysis:", "Let's analyze...", "Let me think", "Analyze User Input", "Determine Appropriate Response", "Draft Response", or numbered planning steps.
   - Begin your response immediately with the direct user-facing message.
2. Verified Evidence ONLY:
   - Answer ONLY from verified image/GIS/tool evidence provided in context.
   - Treat stored evidence, classifications, and GIS metrics as authoritative ground truth.
3. Strict Anti-Hallucination:
   - NEVER invent soil types, coordinates, areas, percentages, or confidence.
   - Do NOT guess satellite parameters, sensors, or dates not provided in context.
4. Missing Evidence:
   - If evidence is missing or context does not contain enough verified evidence to answer reliably, explicitly say: "Insufficient verified evidence to answer reliably."
5. Remove Generic / Template Content:
   - Return one clear, natural, user-facing answer.
   - Remove generic/template content unrelated to the user's question.
   - Do NOT inject generic boilerplate headings such as ### Analysis, ### Key Observations, ### Interpretation, or ### Confidence.
6. Warm & Helpful Persona:
   - When the user introduces themselves (e.g. "my name is shankar"), greet them warmly by name (e.g. "Nice to meet you, Shankar! 👋") and ask how you can help them with satellite imagery, Earth observation, remote sensing, or GIS.
7. Multi-Turn Context & References:
   - Understand and resolve follow-up references such as "that area", "the buildings you mentioned", "the previous image", or "what changed between them".
   - Distinguish multiple images using their image IDs and tasks (e.g. Image A vs Image B, Before vs After, Optical vs SAR).
8. Honest Capability:
   - If no imagery has been analyzed in this session yet, answer general geospatial questions factually while noting that no image context has been uploaded."""


# ---------------------------------------------------------------- reasoning cleaner
DELIBERATION_SENTENCE_PATTERNS = [
    r"^(?:let me|let's|let us)\b",
    r"^(?:maybe\b|perhaps\b)",
    r"^(?:actually\b)",
    r"^(?:i'm not aware\b|i am not aware\b|i'm not sure\b|i am not sure\b|i'm not entirely sure\b|i am not entirely sure\b)",
    r"^(?:i think\b|i believe\b|i assume\b|i suspect\b|we can assume\b)",
    r"^(?:wait,?\s*|okay,?\s*|ok,?\s*|hmm,?\s*)",
    r"^(?:looking at the (?:query|prompt|input|question|image|scene)|looking closely\b)",
    r"^(?:the user is asking|the user wants to know|the user asks|the user might be asking|the user probably wants)\b",
    r"^(?:first,?\s*(?:let's|let me|we should|i should|looking))\b",
    r"^(?:check(?:ing)? against constraints|constraints on output|mental draft|determine appropriate response)\b",
]


def clean_llm_response(text: Optional[str], default_fallback: Optional[str] = None) -> str:
    """
    Strip internal thinking tags, reasoning traces, planning blocks, uncertainty loops,
    and meta-commentary, returning ONLY the clean, final validated user-facing response.
    
    Guarantees:
    - Never exposes 'let me think', 'maybe they mean', 'actually...', 'perhaps...', etc.
    - Strips chain-of-thought XML tags (<think>, <thought>, <reasoning>, <plan>).
    - Preserves legitimate markdown headings, bullet points, and authoritative findings.
    """
    import re
    if not text:
        return default_fallback or ""
    t = str(text).strip()
    if not t:
        return default_fallback or ""

    # 1. Clean outer JSON wrapper if any
    if t.startswith("{") and t.endswith("}"):
        try:
            parsed = json.loads(t)
            if "answer" in parsed:
                t = str(parsed["answer"])
            elif "response" in parsed:
                t = str(parsed["response"])
        except Exception:
            pass

    # 2. Strip XML-style thinking / planning tags: <think>, <thought>, <reasoning>, <plan>, <cot>
    t = re.sub(r"<(think|thought|reasoning|plan|cot|internal_thought)>.*?</\1>", "", t, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # Handle unclosed tag at start of text (e.g. truncated mid-thought <think>...)
    if re.match(r"^<(think|thought|reasoning|plan|cot|internal_thought)>", t, re.IGNORECASE):
        closed_match = re.search(r"</(think|thought|reasoning|plan|cot|internal_thought)>", t, re.IGNORECASE)
        if closed_match:
            t = t[closed_match.end():].strip()
        else:
            return default_fallback or "Hello! How can I assist you with satellite imagery, Earth observation, or GIS today?"

    # 3. Check for planning / reasoning preambles or numbered reasoning sections
    planning_lead_patterns = [
        r"^Here'?s a thinking process:?",
        r"^Here is a thinking process:?",
        r"^Thinking Process:?",
        r"^Thinking:?",
        r"^Let'?s analyze(?:\s+(?:the|this)\s+(?:user\s+)?(?:input|query|prompt|request))?:?",
        r"^Let me think(?:\s+(?:about|through|this))?:?",
        r"^Let me examine(?:\s+(?:the|this))?:?",
        r"^Let us consider(?:\s+(?:the|this))?:?",
        r"^Let'?s consider(?:\s+(?:the|this))?:?",
        r"^Let me check(?:\s+(?:the|this))?:?",
        r"^Let'?s check(?:\s+(?:the|this))?:?",
        r"^(?:(?:\d+\.|\*|-)\s*)?Analyze User Input:?",
        r"^(?:(?:\d+\.|\*|-)\s*)?Analyze Evidence Context:?",
        r"^(?:(?:\d+\.|\*|-)\s*)?Check against constraints:?",
        r"^(?:(?:\d+\.|\*|-)\s*)?Determine (?:the )?(?:Appropriate )?(?:Response|Answer):?",
        r"^(?:(?:\d+\.|\*|-)\s*)?Identify (?:Key Evidence|Role/Persona):?",
        r"^(?:(?:\d+\.|\*|-)\s*)?Extract Relevant Information:?",
        r"^The user is asking:?",
        r"^The prompt (?:says|asks):?",
        r"^Mental Draft:?",
    ]
    
    numbered_reasoning_start = bool(re.match(r"^\s*1\.\s*\*{0,2}(?:Analyze|Identify|Determine|Check|Review|Understand)\b", t, re.IGNORECASE))
    has_planning_lead = any(re.search(pat, t, re.IGNORECASE) for pat in planning_lead_patterns) or numbered_reasoning_start

    if has_planning_lead:
        # Search for handover markers where the model transitions from planning to the draft/final answer
        handover_patterns = [
            r"(?:(?:\d+\.|\*|-)\s*)?\*{0,2}(?:Draft Response|Drafted Response|Final Answer|Final Response|Response to (?:the )?User|Direct Response)\*{0,2}\s*:?\s*",
            r"\n\n+(?=(?:###|\*\*Analysis\*\*|Hello\b|Hi\b|Nice to meet|Greetings\b|Welcome\b|Satellite scene|Based on|Quantitatively|The |Joint |Spectral |Between |Buildings |In this |Visible |Detected ))",
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

    # 4. Strip conversational deliberation / uncertainty loops from beginning of answer
    # Models often deliberate out loud: "Let me think. Maybe they mean... Actually... The buildings are..."
    paragraphs = t.split("\n\n")
    cleaned_paragraphs = []
    dropping_deliberation = True

    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue

        # Check if the paragraph starts with markdown heading or list item — always keep those
        if p_clean.startswith("###") or p_clean.startswith("##") or p_clean.startswith("- ") or p_clean.startswith("* "):
            dropping_deliberation = False
            cleaned_paragraphs.append(p_clean)
            continue

        if dropping_deliberation:
            # Check if this paragraph is purely deliberative
            is_deliberative_para = False
            for pat in DELIBERATION_SENTENCE_PATTERNS:
                if re.match(pat, p_clean, re.IGNORECASE):
                    is_deliberative_para = True
                    break

            if is_deliberative_para:
                # See if there is a direct answer embedded after deliberation in this paragraph
                sentences = re.split(r"(?<=[.!?])\s+", p_clean)
                valid_sentences = []
                found_non_delib = False
                for s in sentences:
                    s_str = s.strip()
                    if not s_str:
                        continue
                    is_s_delib = any(re.match(pat, s_str, re.IGNORECASE) for pat in DELIBERATION_SENTENCE_PATTERNS)
                    if not found_non_delib and is_s_delib:
                        continue
                    found_non_delib = True
                    valid_sentences.append(s_str)

                if valid_sentences:
                    dropping_deliberation = False
                    cleaned_paragraphs.append(" ".join(valid_sentences))
                # Else this entire paragraph was deliberation; drop it
                continue
            else:
                dropping_deliberation = False
                cleaned_paragraphs.append(p_clean)
        else:
            cleaned_paragraphs.append(p_clean)

    if cleaned_paragraphs:
        t = "\n\n".join(cleaned_paragraphs).strip()

    # 5. Final pass for leftover lead phrases
    t = re.sub(r"^Here'?s a thinking process:\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^Thinking Process:\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^Thinking:\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^(?:(?:Drafted\s+)?Response|Final\s+Answer|Direct\s+Response):\s*", "", t, flags=re.IGNORECASE).strip()

    # 6. Strip canned template boilerplate sections
    canned_boilerplate_patterns = [
        r"### Interpretation\s*\n\s*Spatial analysis reflects surface land-cover distribution across the footprint\.?",
        r"### Interpretation\s*\n\s*Spatial vectorization and temporal comparison delineate localized transformation across observation dates\.?",
        r"### Interpretation\s*\n\s*Integrating microwave radar backscatter with optical reflectance clarifies structural footprints and standing water\.?",
        r"### Interpretation\s*\n\s*Land-cover distribution differentiates natural vegetation, agricultural zones, and built surfaces\.?",
        r"### Confidence\s*\n\s*-\s*\*\*Status\*\*:\s*Empirical confidence score:\s*\*\*[\d\.]+\*\*[\s\S]*?(?=\n\n|\Z)",
    ]
    for cbp in canned_boilerplate_patterns:
        t = re.sub(cbp, "", t, flags=re.IGNORECASE).strip()

    # If t is wrapped in JSON e.g. {"answer": "..."}
    if (t.startswith("{") and "}" in t) or ("```" in t):
        try:
            from ai.llm.base import LLMResponse
            dummy_resp = LLMResponse(content=t, model="", provider="", latency_ms=0)
            parsed = dummy_resp.json()
            if isinstance(parsed, dict) and ("answer" in parsed or "response" in parsed):
                extracted = parsed.get("answer") or parsed.get("response")
                if extracted and isinstance(extracted, str):
                    t = extracted.strip()
        except Exception:
            pass

    # If ### Analysis is the only section heading on the entire answer, strip it to return one clear, natural answer
    if t.startswith("### Analysis\n") and "### " not in t[13:]:
        t = t[13:].strip()

    # If everything was stripped because it was entirely internal reasoning
    if not t or len(t.strip()) < 3:
        return default_fallback or "Hello! How can I assist you with satellite imagery, Earth observation, or GIS today?"

    return t.strip()


def clean_chat_response(text: Optional[str]) -> str:
    """Backward-compatible alias for clean_llm_response."""
    return clean_llm_response(text)


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
