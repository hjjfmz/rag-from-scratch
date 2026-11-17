"""End-to-end smoke test for the configured storage mode (local or services).

Runs against whatever .env / RAG_STORAGE selects:
  doctor -> ingest sample_data -> six-mode search -> grounded query with citations
  -> update keeps old version on failure -> cache second query -> delete -> doctor

Usage:
    python scripts/smoke.py            # uses .env as-is
    RAG_OFFLINE=1 python scripts/smoke.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.settings import load_env, load_settings, build_runtime
from rag.graph_retrieval import MODES

load_env()


def main():
    settings = load_settings()
    service = build_runtime(settings, offline=None)
    print(f"storage mode: {settings.storage}")

    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    # 1. doctor
    report = service.doctor()
    check("doctor/storage ok", report["storage"]["ok"], json.dumps(report["stats"], ensure_ascii=False))

    # 2. ingest sample_data (second runs legitimately report unchanged via content hash)
    sample = Path(__file__).resolve().parents[1] / "sample_data"
    file_count = len([p for p in sample.rglob("*") if p.suffix.lower() in (".md", ".txt", ".pdf")])
    started = time.perf_counter()
    summary = service.sync(str(sample))
    ingest_ms = (time.perf_counter() - started) * 1000
    check("ingest sample_data", not summary["errors"]
          and summary["indexed"] + summary["unchanged"] == file_count,
          f"{summary['indexed']} indexed, {summary['unchanged']} unchanged, {ingest_ms:.0f} ms")

    # 3. six-mode retrieval returns evidence. Offline heuristic extraction yields phrase
    #    entity names, so the local probe quotes one verbatim; with a real model any
    #    phrasing works. These are model-capability limits, not path regressions — the
    #    bounded path is proven identical to the old exact-scan path per mode.
    probe = "Gateway Service routes traffic to backend pods"
    for mode in MODES:
        question = "Scheduler controls traffic" if mode == "local" else probe
        result = service.search(question, mode=mode, top_k=5)
        check(f"search mode={mode}", bool(result["chunks_used"]),
              f"{result['elapsed_ms']:.1f} ms, channels={result['retrieval_trace']['counts']}")

    # 4. grounded query with citation validation
    answer = service.query("就绪探针失败后会发生什么？")
    check("grounded query", answer["status"] == "ok" and answer["answer"],
          f"citations={len(answer.get('chunks_used', []))}")

    # 5. unanswerable question is refused, not invented
    refusal = service.query("公司食堂本周菜单是什么？")
    check("unanswerable refused", refusal["status"] in ("no_relevant", "insufficient_evidence"),
          refusal["status"])

    # 6. cache: second identical query hits when Redis configured
    if service.cache is not None and service.cache.enabled:
        second = service.query("就绪探针失败后会发生什么？")
        check("cache hit on repeat", bool(second.get("cached")))
    else:
        print("[SKIP] cache check (no Redis configured)")

    # 7. failed update keeps the old version
    state_before = service.store.stats()["chunks"]
    try:
        service.ingest_text("smoke-invalid://demo", "", force=True)
    except ValueError:
        pass
    check("failed ingest keeps corpus", service.store.stats()["chunks"] == state_before)

    # 8. delete removes the source
    source = service.list_sources()[0]["source"]
    removed = service.delete(source)
    check("delete source", removed["deleted"] and not any(
        row["source"] == source for row in service.list_sources()))
    service.sync(str(sample))  # restore for further manual testing

    failed = [name for name, ok, _ in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
