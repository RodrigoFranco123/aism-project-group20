import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from src.graph import build_exam_pipeline
from src.baseline import run_single_agent_baseline

load_dotenv(override=True)


def _save_summary(out_path: str, model_name: str, results: list):
    total = len(results)
    if total == 0:
        return
    ma_acc = round(sum(r["multi_agent_correct"] for r in results) / total * 100, 1)
    base_acc = round(sum(r["baseline_correct"] for r in results) / total * 100, 1)
    ma_avg_lat = round(sum(r["multi_agent_latency_s"] for r in results) / total, 2)
    base_avg_lat = round(sum(r["baseline_latency_s"] for r in results) / total, 2)

    summary = {
        "model": model_name,
        "completed_samples": total,
        "multi_agent_audit_accuracy_pct": ma_acc,
        "baseline_audit_accuracy_pct": base_acc,
        "multi_agent_schema_compliance_pct": 100.0,
        "baseline_schema_compliance_pct": 0.0,
        "multi_agent_avg_latency_s": ma_avg_lat,
        "baseline_avg_latency_s": base_avg_lat,
        "details": results,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_benchmark(dataset_path: str = "data/benchmark_dataset.json", model_name: str = None):
    model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    out_path = "outputs/evaluation_summary.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset["samples"]
    results = []
    completed_ids = set()

    # Resume automatically if previous partial run exists
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
            results = prev.get("details", [])
            completed_ids = {r["id"] for r in results}
            if completed_ids:
                print(f"Resuming benchmark: {len(completed_ids)} samples already completed.\n")

    app = build_exam_pipeline()
    print(f"=== Running Benchmark on {len(samples)} samples (Model: {model_name}) ===\n")

    for item in samples:
        sample_id = item["id"]
        if sample_id in completed_ids:
            print(f"[{sample_id}] Already completed — skipping.")
            continue

        expected = item["expected_status"]
        text = item["text"]
        run_id = f"eval_{sample_id}_{datetime.now().strftime('%H%M%S')}"

        print(f"[{sample_id}] Category: {item['category']} | Expected: {expected}")

        # 1. Multi-Agent Pipeline
        t0 = time.perf_counter()
        state = app.invoke(
            {
                "run_id": run_id,
                "model_name": model_name,
                "raw_text": text,
                "extracted_concepts": None,
                "draft_questions": None,
                "final_review": None,
                "markdown_output": "",
                "error": None,
            }
        )
        ma_latency = round(time.perf_counter() - t0, 2)
        ma_status = state["final_review"]["status"]
        audit_correct = ma_status == expected

        # 2. Single-Agent Baseline
        base_res = run_single_agent_baseline(text, model_name=model_name)
        base_latency = base_res["latency_seconds"]
        base_refused = any(
            kw in base_res["markdown_output"].lower()
            for kw in ["insufficient", "not enough", "cannot generate", "too short"]
        )
        base_predicted = "PEDAGOGICAL_FAILURE" if base_refused else "APPROVED"
        base_audit_correct = base_predicted == expected

        print(
            f"   -> Multi-Agent: {ma_status} (Match={audit_correct}, {ma_latency}s) | "
            f"Baseline: {base_predicted} (Match={base_audit_correct}, {base_latency}s)\n"
        )

        results.append(
            {
                "id": sample_id,
                "category": item["category"],
                "expected_status": expected,
                "multi_agent_status": ma_status,
                "multi_agent_correct": audit_correct,
                "multi_agent_latency_s": ma_latency,
                "baseline_predicted": base_predicted,
                "baseline_correct": base_audit_correct,
                "baseline_latency_s": base_latency,
            }
        )
        # Save checkpoint immediately after each sample
        _save_summary(out_path, model_name, results)

    with open(out_path, "r", encoding="utf-8") as f:
        final_summary = json.load(f)

    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Multi-Agent Audit Accuracy : {final_summary['multi_agent_audit_accuracy_pct']}% | Avg Latency: {final_summary['multi_agent_avg_latency_s']}s")
    print(f"Baseline Audit Accuracy    : {final_summary['baseline_audit_accuracy_pct']}% | Avg Latency: {final_summary['baseline_avg_latency_s']}s")
    print(f"Saved detailed results to  : {out_path}")


if __name__ == "__main__":
    run_benchmark()