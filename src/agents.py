import json
import os
import time
from datetime import datetime, timezone
from typing import TypedDict, Optional, Type, TypeVar
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from src.schemas import (
    ConceptExtractionOutput,
    QuestionGenerationOutput,
    PedagogicalReviewOutput,
)
from src.prompts import (
    CONCEPT_EXTRACTOR_PROMPT,
    QUESTION_WRITER_PROMPT,
    PEDAGOGICAL_REVIEWER_PROMPT,
)

T = TypeVar("T", bound=BaseModel)


class PipelineState(TypedDict):
    run_id: str
    model_name: str
    raw_text: str
    extracted_concepts: Optional[dict]
    draft_questions: Optional[dict]
    final_review: Optional[dict]
    markdown_output: str
    error: Optional[str]


def _log_event(run_id: str, agent_name: str, payload: dict, latency_sec: float):
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"trace_{run_id}.jsonl")
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "agent": agent_name,
        "latency_seconds": round(latency_sec, 3),
        "payload": payload,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _invoke_validated_json(
    model_name: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    schema_cls: Type[T],
    max_retries: int = 3,
) -> T:
    llm = ChatOllama(model=model_name, temperature=temperature, format="json")
    last_err = None
    prompt_to_send = user_prompt

    for attempt in range(1, max_retries + 1):
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt_to_send),
            ]
        )
        raw_content = response.content.strip()
        try:
            data = json.loads(raw_content)
            return schema_cls.model_validate(data)
        except Exception as e:
            last_err = e
            print(f"   [Retry {attempt}/{max_retries}] Schema validation adjustment...")
            prompt_to_send = (
                f"{user_prompt}\n\n"
                f"PREVIOUS OUTPUT FAILED VALIDATION WITH ERROR:\n{e}\n"
                "Fix the JSON structure so it strictly satisfies all counts and field names."
            )

    raise RuntimeError(f"Failed to produce valid {schema_cls.__name__} after {max_retries} attempts: {last_err}")


def concept_extractor_node(state: PipelineState) -> dict:
    print("-> [Agent 1: Concept Extractor] Extracting 5 core concepts...")
    start = time.perf_counter()
    result = _invoke_validated_json(
        model_name=state["model_name"],
        temperature=0.1,
        system_prompt=CONCEPT_EXTRACTOR_PROMPT,
        user_prompt=f"SOURCE MATERIAL:\n{state['raw_text']}",
        schema_cls=ConceptExtractionOutput,
    )
    elapsed = time.perf_counter() - start
    payload = result.model_dump()
    _log_event(state["run_id"], "Agent1_ConceptExtractor", payload, elapsed)
    print(f"   [Agent 1] Done ({elapsed:.1f}s)")
    return {"extracted_concepts": payload}


def question_writer_node(state: PipelineState) -> dict:
    print("-> [Agent 2: Question Writer] Generating 5 MCQs and 3 flashcards...")
    start = time.perf_counter()
    concepts_json = json.dumps(state["extracted_concepts"], indent=2, ensure_ascii=False)
    result = _invoke_validated_json(
        model_name=state["model_name"],
        temperature=0.3,
        system_prompt=QUESTION_WRITER_PROMPT,
        user_prompt=f"EXTRACTED CONCEPTS JSON:\n{concepts_json}",
        schema_cls=QuestionGenerationOutput,
    )
    elapsed = time.perf_counter() - start
    payload = result.model_dump()
    _log_event(state["run_id"], "Agent2_QuestionWriter", payload, elapsed)
    print(f"   [Agent 2] Done ({elapsed:.1f}s)")
    return {"draft_questions": payload}


def pedagogical_reviewer_node(state: PipelineState) -> dict:
    print("-> [Agent 3: Pedagogical Reviewer] Auditing questions and grounding...")
    start = time.perf_counter()

    # Deterministic pre-audit check for severe source insufficiency / circularity
    words = state["raw_text"].split()
    sentences = [s.strip() for s in state["raw_text"].split(".") if len(s.strip()) > 3]
    if len(words) < 50 or len(sentences) < 4:
        elapsed = time.perf_counter() - start
        payload = {
            "status": "PEDAGOGICAL_FAILURE",
            "failure_reason": (
                f"Insufficient Material / Pedagogical Failure: Source text contains only "
                f"{len(words)} words across {len(sentences)} sentences, forcing Agent 1 to "
                f"hallucinate peripheral concepts to meet the 5-concept quota."
            ),
            "reviewed_mcqs": [],
            "approved_flashcards": [],
        }
        _log_event(state["run_id"], "Agent3_PedagogicalReviewer", payload, elapsed)
        print(f"   [Agent 3] Done ({elapsed:.2f}s) -> Status: PEDAGOGICAL_FAILURE")
        return {"final_review": payload, "markdown_output": render_markdown(payload)}

    draft_json = json.dumps(state["draft_questions"], indent=2, ensure_ascii=False)
    audit_input = (
        f"ORIGINAL SOURCE MATERIAL:\n{state['raw_text']}\n\n"
        f"DRAFT EXAM & FLASHCARDS JSON:\n{draft_json}"
    )
    result = _invoke_validated_json(
        model_name=state["model_name"],
        temperature=0.0,
        system_prompt=PEDAGOGICAL_REVIEWER_PROMPT,
        user_prompt=audit_input,
        schema_cls=PedagogicalReviewOutput,
    )
    elapsed = time.perf_counter() - start
    payload = result.model_dump()
    _log_event(state["run_id"], "Agent3_PedagogicalReviewer", payload, elapsed)
    print(f"   [Agent 3] Done ({elapsed:.1f}s) -> Status: {payload['status']}")
    md = render_markdown(payload)
    return {"final_review": payload, "markdown_output": md}


def render_markdown(review: dict) -> str:
    if review["status"] == "PEDAGOGICAL_FAILURE":
        return (
            "# AUDIT ALERT: INSUFFICIENT MATERIAL / PEDAGOGICAL FAILURE\n\n"
            f"**Reason:** {review['failure_reason']}\n\n"
            "The Pedagogical Reviewer halted artifact generation to prevent invalid or hallucinated assessment items."
        )

    lines = ["# Generated Exam & Study Flashcards\n", "## Multiple-Choice Questions\n"]
    for idx, mcq in enumerate(review["reviewed_mcqs"], 1):
        lines.append(f"### Q{idx}. {mcq['question']}")
        for opt in mcq["options"]:
            lines.append(f"- {opt}")
        lines.append(f"\n**Correct Answer:** {mcq['correct_option']}")
        lines.append("**Distractor Analysis:**")
        for just in mcq["distractor_justifications"]:
            lines.append(f"  - {just}")
        lines.append("")

    lines.append("## Active Recall Flashcards\n")
    for idx, fc in enumerate(review["approved_flashcards"], 1):
        lines.append(f"**Card {idx} — Front:** {fc['front']}")
        lines.append(f"**Card {idx} — Back:** {fc['back']}\n")

    return "\n".join(lines)