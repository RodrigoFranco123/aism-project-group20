import time
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


def run_single_agent_baseline(raw_text: str, model_name: str = "llama3.1:8b") -> dict:
    start = time.perf_counter()
    llm = ChatOllama(model=model_name, temperature=0.2)
    prompt = (
        "Extract 5 key concepts from the text below, generate 5 multiple-choice questions "
        "(4 options each, with correct answer and distractor explanations) and 3 flashcards "
        "in Markdown. If the text is insufficient, warn the user."
    )
    response = llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=raw_text),
        ]
    )
    elapsed = round(time.perf_counter() - start, 3)
    return {"latency_seconds": elapsed, "markdown_output": response.content}