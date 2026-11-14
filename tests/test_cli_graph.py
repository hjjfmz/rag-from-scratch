"""Run the actual command line across process restarts, not only parser helpers."""
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_cli(tmp_path, *args, ok=True):
    env = {**os.environ, "PYTHONUTF8": "1", "TEMP": str(ROOT / ".runtime/tmp"), "TMP": str(ROOT / ".runtime/tmp")}
    result = subprocess.run([sys.executable, str(ROOT / "cli.py"), *args, "--offline", "--json", "--data-dir", str(tmp_path / "data")],
                            cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=40)
    if ok:
        assert result.returncode == 0, result.stderr + result.stdout
    else:
        assert result.returncode != 0
    return json.loads(result.stdout)


def test_cli_complete_persistent_graph_workflow(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("Pod uses Container.\nService routes to Pod.", encoding="utf-8")
    assert run_cli(tmp_path, "ingest", "--source", str(source))["relations"] == 2
    fine = run_cli(tmp_path, "search", "Pod", "--search-mode", "local")
    assert fine["relations"]
    assert fine["retrieval_trace"]["local"]["seed_entities"]
    coarse = run_cli(tmp_path, "query", "Service routes", "--search-mode", "global")
    assert "[S1]" in coarse["answer"]
    assert run_cli(tmp_path, "graph")["relations"]
    assert run_cli(tmp_path, "doctor")["storage"]["ok"] is True
    assert run_cli(tmp_path, "delete", "--source", str(source))["deleted"] is True
    assert run_cli(tmp_path, "query", "Pod")["status"] == "no_docs"


def test_cli_invalid_limits_are_structured_errors(tmp_path):
    result = run_cli(tmp_path, "query", "Pod", "--top-k", "0", ok=False)
    assert result["status"] == "error"


def test_cli_cross_file_evidence_and_user_supplied_evaluation(tmp_path):
    first, second = tmp_path / "project.md", tmp_path / "storage.md"
    first.write_text("Project uses Pipeline.", encoding="utf-8")
    second.write_text("Pipeline uses SQLite. SQLite provides AtomicCommit.", encoding="utf-8")
    run_cli(tmp_path, "ingest", "--source", str(first))
    run_cli(tmp_path, "ingest", "--source", str(second))
    result = run_cli(tmp_path, "query", "Project Pipeline SQLite", "--top-k", "2")
    assert {Path(source).name for source in result["sources"]} == {"project.md", "storage.md"}
    assert {r["citation_id"] for r in result["relations"]} == {"S1", "S2"}
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": "one", "question": "Project Pipeline SQLite",
                                    "expected_sources": ["project.md", "storage.md"], "answerable": True}]), encoding="utf-8")
    evaluated = run_cli(tmp_path, "eval", "--test-file", str(questions))
    assert evaluated["num_questions"] == 1
    assert evaluated["summary"]["hybrid_graph"]["source_recall"] == 1.


def test_cli_model_absence_never_silently_enables_offline(tmp_path):
    env = {key: value for key, value in os.environ.items() if key not in
           ("LLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "CHAT_API_KEY", "EMBEDDING_API_KEY")}
    env["PYTHONUTF8"] = "1"
    env["RAG_ENV_FILE"] = str(tmp_path / "intentionally-missing.env")
    result = subprocess.run([sys.executable, str(ROOT / "cli.py"), "query", "Pod", "--json", "--data-dir", str(tmp_path / "data")],
                            cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert result.returncode != 0
    assert json.loads(result.stdout)["status"] == "error"


def test_cli_outputs_match_actual_evidence_after_update(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("Pod uses OldContainer.", encoding="utf-8")
    run_cli(tmp_path, "ingest", "--source", str(source))
    source.write_text("Pod uses NewContainer.", encoding="utf-8")
    run_cli(tmp_path, "ingest", "--source", str(source))
    result = run_cli(tmp_path, "query", "Pod", "--search-mode", "local")
    assert "NewContainer" in result["answer"]
    assert "OldContainer" not in result["answer"]
