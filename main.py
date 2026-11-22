import argparse
import os
from datetime import datetime
from dotenv import load_dotenv
from src.graph import build_exam_pipeline
from src.baseline import run_single_agent_baseline

load_dotenv(override=True)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Exam & Flashcard Generator (Group 20)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample_notes_valid.txt",
        help="Path to input notes",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
        help="Ollama model name",
    )
    parser.add_argument(
        "--mode",
        choices=["multi-agent", "baseline"],
        default="multi-agent",
        help="Execution mode: multi-agent pipeline or single-agent baseline",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        raise ValueError(f"Input file is empty: {args.input}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs", exist_ok=True)

    if args.mode == "baseline":
        print(f"[Baseline] Running single-agent prompt with {args.model}...")
        res = run_single_agent_baseline(raw_text, args.model)
        out_file = f"outputs/baseline_{run_id}.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(res["markdown_output"])
        print(f"Done in {res['latency_seconds']}s -> Saved to {out_file}")
        return

    print(f"[Multi-Agent Pipeline] Run ID: {run_id} | Model: {args.model}")
    app = build_exam_pipeline()
    final_state = app.invoke(
        {
            "run_id": run_id,
            "model_name": args.model,
            "raw_text": raw_text,
            "extracted_concepts": None,
            "draft_questions": None,
            "final_review": None,
            "markdown_output": "",
            "error": None,
        }
    )

    out_file = f"outputs/exam_{run_id}.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(final_state["markdown_output"])

    status = final_state["final_review"]["status"]
    print(f"Pipeline finished with status: {status}")
    print(f"Forensic trace log: logs/trace_{run_id}.jsonl")
    print(f"Generated artifact: {out_file}")


if __name__ == "__main__":
    main()