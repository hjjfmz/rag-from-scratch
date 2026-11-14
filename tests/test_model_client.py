"""Model boundary tests use real HTTP locally and never contact a paid API."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rag.model_client import ModelClient, ModelError


@pytest.fixture
def remote(monkeypatch):
    state = {"calls": [], "active": 0, "peak": 0}
    lock = threading.Lock()

    def default(path, payload, headers):
        return 200, {"data": [{"index": i, "embedding": [1.0, float(i)]}
                              for i, _ in enumerate(payload.get("input", []))]}, {}, 0

    state["respond"] = default

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["calls"].append((self.path, payload, dict(self.headers), time.monotonic()))
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                status, body, headers, delay = state["respond"](self.path, payload, self.headers)
                time.sleep(delay)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                try:
                    self.wfile.write((body if isinstance(body, str) else json.dumps(body)).encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass
            finally:
                with lock:
                    state["active"] -= 1

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for key in ("CHAT_API_KEY", "EMBEDDING_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    root = f"http://127.0.0.1:{server.server_port}"
    for key, value in {"CHAT_API_KEY": "chat-secret", "EMBEDDING_API_KEY": "embed-secret",
                       "CHAT_API_URL": root + "/chat/completions", "EMBEDDING_API_URL": root + "/embeddings",
                       "MODEL_CHAT": "chat-test", "MODEL_EMBEDDING": "embed-test",
                       "MODEL_TIMEOUT_SECONDS": "0.1", "MODEL_MAX_RETRIES": "1",
                       "MODEL_MAX_CONCURRENCY": "2", "MODEL_REQUESTS_PER_SECOND": "1000"}.items():
        monkeypatch.setenv(key, value)
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def chat(body):
    return 200, {"choices": [{"message": {"content": json.dumps(body)}}]}, {}, 0


def test_missing_configuration_never_silently_uses_offline(monkeypatch):
    for key in ("CHAT_API_KEY", "EMBEDDING_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ModelError, match="config|key|KEY"):
        ModelClient()


def test_429_retry_after_then_reorders_embeddings(remote):
    def respond(*_):
        if len(remote["calls"]) == 1:
            return 429, {"error": "busy"}, {"Retry-After": "0.03"}, 0
        return 200, {"data": [{"index": 1, "embedding": [0, 1]},
                              {"index": 0, "embedding": [1, 0]}]}, {}, 0
    remote["respond"] = respond
    client = ModelClient()
    assert client.embed(["alpha", "beta"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert len(remote["calls"]) == 2
    assert remote["calls"][1][3] - remote["calls"][0][3] >= 0.025
    assert client.metrics["retries"] == 1


def test_timeout_retries_exhausted_as_model_error(remote):
    remote["respond"] = lambda *_: (200, {"data": []}, {}, 0.3)
    with pytest.raises(ModelError, match="timeout|request|failed"):
        ModelClient().embed(["slow"])
    assert len(remote["calls"]) == 2


@pytest.mark.parametrize("data", [
    [{"index": 0, "embedding": [True, 1]}],
    [{"index": 0, "embedding": [float("nan"), 1]}],
    [{"index": 0, "embedding": ["1", 0]}],
    [{"index": 0, "embedding": []}],
    [{"index": 1, "embedding": [1, 0]}],
    [{"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [1, 0]}],
])
def test_embedding_rejects_malformed_vectors(remote, data):
    remote["respond"] = lambda *_: (200, {"data": data}, {}, 0)
    with pytest.raises(ModelError):
        ModelClient().embed(["text"])


def test_embedding_dimension_is_consistent_across_calls(remote):
    client = ModelClient()
    client.embed(["one"])
    remote["respond"] = lambda *_: (200, {"data": [{"index": 0, "embedding": [1, 2, 3]}]}, {}, 0)
    with pytest.raises(ModelError, match="dimension"):
        client.embed(["two"])


def test_bad_json_and_permanent_http_failure_are_not_retried(remote):
    remote["respond"] = lambda *_: (200, "not json", {}, 0)
    with pytest.raises(ModelError):
        ModelClient().embed(["text"])
    assert len(remote["calls"]) == 1
    remote["respond"] = lambda *_: (401, {"error": "bad key"}, {}, 0)
    with pytest.raises(ModelError):
        ModelClient().embed(["text"])
    assert len(remote["calls"]) == 2


def test_cache_survives_reopen_and_excludes_secrets(remote, tmp_path, monkeypatch):
    path = tmp_path / "cache.sqlite"
    first = ModelClient(cache_path=path)
    assert first.embed(["alpha"]) == [[1, 0]]
    second = ModelClient(cache_path=path)
    assert second.embed(["alpha"]) == [[1, 0]]
    assert len(remote["calls"]) == 1
    assert second.metrics["cache_hits"] == 1
    assert first.fingerprint == second.fingerprint
    monkeypatch.setenv("EMBEDDING_API_KEY", "different-secret")
    assert ModelClient().fingerprint == first.fingerprint
    monkeypatch.setenv("MODEL_EMBEDDING", "different-model")
    changed = ModelClient(cache_path=path)
    assert changed.fingerprint != first.fingerprint
    changed.embed(["alpha"])
    assert len(remote["calls"]) == 2
    assert b"embed-secret" not in path.read_bytes()


def test_concurrency_is_bounded_independently_of_rate(remote, monkeypatch):
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "1")
    remote["respond"] = lambda *_: (200, {"data": [{"index": 0, "embedding": [1, 0]}]}, {}, 0.08)
    client = ModelClient()
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert len(list(pool.map(lambda i: client.embed([str(i)]), range(6)))) == 6
    assert remote["peak"] <= 2


def test_request_rate_is_bounded(remote, monkeypatch):
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("MODEL_REQUESTS_PER_SECOND", "10")
    remote["respond"] = lambda *_: (200, {"data": [{"index": 0, "embedding": [1, 0]}]}, {}, 0)
    client = ModelClient()
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert len(list(pool.map(lambda i: client.embed([str(i)]), range(6)))) == 6
    starts = sorted(call[3] for call in remote["calls"])
    # Server arrival can coalesce under Windows scheduling; check the rate over
    # the batch instead of assuming dispatch and arrival have identical timing.
    assert starts[-1] - starts[0] >= 0.35


def test_separate_chat_and_embedding_keys(remote):
    client = ModelClient()
    client.embed(["Pod"])
    remote["respond"] = lambda *_: chat({"low_level": ["Pod"], "high_level": ["routing"]})
    assert client.keywords("Pod routing")["low_level"] == ["Pod"]
    assert remote["calls"][0][2]["Authorization"] == "Bearer embed-secret"
    assert remote["calls"][1][2]["Authorization"] == "Bearer chat-secret"


def test_extract_validates_original_evidence_and_endpoints(remote):
    valid = {"entities": [{"name": "Pod", "type": "resource", "description": "workload"},
                          {"name": "Container", "type": "resource", "description": "runtime"}],
             "relations": [{"source": "Pod", "target": "Container", "relation": "uses",
                            "description": "Pod uses Container", "keywords": ["uses"],
                            "evidence": "Pod uses Container."}]}
    remote["respond"] = lambda *_: chat(valid)
    assert ModelClient().extract("Pod uses Container.") == valid
    valid["relations"][0]["evidence"] = "Pod uses imaginary things."
    with pytest.raises(ModelError, match="evidence"):
        ModelClient().extract("Pod uses Container.")
    valid["relations"][0]["evidence"] = "Pod uses Container."
    valid["relations"][0]["target"] = "Missing"
    with pytest.raises(ModelError, match="endpoint|entity"):
        ModelClient().extract("Pod uses Container.")


def test_remote_plan_rerank_and_answer_validate_structures(remote):
    client = ModelClient()
    remote["respond"] = lambda *_: chat(["Pod restarts", "Probe failure"])
    assert client.plan("why restart?") == ["Pod restarts", "Probe failure"]
    contexts = [{"citation_id": "S1", "text": "Pod uses Container.", "chunk_id": "a"},
                {"citation_id": "S2", "text": "Probe checks health.", "chunk_id": "b"}]
    remote["respond"] = lambda *_: chat({"scores": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.2}]})
    ranked = client.rerank("health", contexts)
    assert ranked[0] == {**contexts[1], "rerank_score": 0.9}
    remote["respond"] = lambda *_: (200, {"choices": [{"message": {"content": "Pod uses Container. [S1]"}}]}, {}, 0)
    assert client.answer("What?", contexts, []) == "Pod uses Container. [S1]"
    remote["respond"] = lambda *_: (200, {"choices": [{"message": {"content": "Unsupported [S99]"}}]}, {}, 0)
    with pytest.raises(ModelError, match="citation"):
        client.answer("What?", contexts, [])


def test_offline_is_deterministic_and_keeps_literal_evidence():
    client = ModelClient(offline=True)
    text = "Pod uses a Container. Probe restarts Container. Service routes to Pod. Deployment manages ReplicaSet."
    graph = client.extract(text)
    assert {r["relation"] for r in graph["relations"]} == {"uses", "restarts", "routes to", "manages"}
    assert all(r["evidence"] in text for r in graph["relations"])
    assert {e["name"] for e in graph["entities"]} >= {"Pod", "Container", "Probe", "Service", "Deployment", "ReplicaSet"}
    assert graph == ModelClient(offline=True).extract(text)
    vectors = client.embed(["Pod uses Container", "Pod uses Container", "服务连接容器"])
    assert vectors[0] == vectors[1]
    assert all(len(v) == len(vectors[0]) for v in vectors)
    assert any(vectors[2])
    assert client.fingerprint == ModelClient(offline=True).fingerprint
    contexts = [{"citation_id": "S1", "text": "Pod uses a Container."}]
    answer = client.answer("What does Pod use?", contexts, graph["relations"])
    assert "offline" in answer.lower() and "[S1]" in answer
    assert "Pod uses a Container." in answer
    assert client.health()["offline"] is True
    assert client.keywords("Pod 故障诊断")["low_level"]
    assert client.plan("Pod restart")
    assert "rerank_score" in client.rerank("Pod", contexts)[0]


@pytest.mark.parametrize("verb", ["requires", "selects", "contains", "exposes"])
def test_offline_extract_general_relations_not_query_specific(verb):
    text = f"Alpha {verb} a Beta."
    relation = ModelClient(offline=True).extract(text)["relations"][0]
    assert (relation["source"], relation["target"], relation["relation"]) == ("Alpha", "Beta", verb)
    assert relation["evidence"] == text


@pytest.mark.parametrize("method,args", [("embed", [[1]]), ("extract", [None]),
                                        ("keywords", [""]), ("plan", [""])])
def test_invalid_public_inputs_fail_explicitly(method, args):
    with pytest.raises(ModelError):
        getattr(ModelClient(offline=True), method)(*args)


def test_overflowing_vector_component_is_model_error(remote):
    remote["respond"] = lambda *_: (200, {"data": [{"index": 0, "embedding": [10 ** 400]}]}, {}, 0)
    with pytest.raises(ModelError):
        ModelClient().embed(["overflow"])


def test_non_json_context_metadata_is_model_error():
    with pytest.raises(ModelError):
        ModelClient(offline=True).rerank("Pod", [{"text": "Pod runs Container.", "bad": object()}])


def test_generic_key_priority_and_no_network_for_refusal(remote, monkeypatch):
    monkeypatch.delenv("CHAT_API_KEY")
    monkeypatch.delenv("EMBEDDING_API_KEY")
    monkeypatch.setenv("LLM_API_KEY", "primary-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "secondary-secret")
    client = ModelClient()
    assert "No supporting evidence" in client.answer("Unknown?", [], [])
    assert remote["calls"] == []
    client.embed(["Pod"])
    assert remote["calls"][0][2]["Authorization"] == "Bearer primary-secret"


@pytest.mark.parametrize("task,result", [
    ("keywords", {"low_level": "Pod", "high_level": []}),
    ("plan", {"queries": []}),
    ("plan", {"queries": ["x"] * 5}),
    ("extract", {"entities": [], "relations": "bad"}),
])
def test_remote_task_schema_rejects_invalid_json_structures(remote, task, result):
    remote["respond"] = lambda *_: chat(result)
    with pytest.raises(ModelError):
        getattr(ModelClient(), task)("Pod")


def test_invalid_rerank_does_not_drop_or_duplicate_contexts(remote):
    remote["respond"] = lambda *_: chat({"scores": [{"index": 0, "score": 0.9}, {"index": 0, "score": 0.1}]})
    with pytest.raises(ModelError):
        ModelClient().rerank("Pod", [{"text": "Pod"}, {"text": "Service"}])


def test_offline_health_ignores_invalid_remote_config(monkeypatch):
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "broken")
    assert ModelClient(offline=True).health()["offline"] is True


def test_remote_embedding_batches_130_inputs_and_preserves_global_order(remote, tmp_path):
    def respond(path, payload, headers):
        return 200, {"data": [{"index": i, "embedding": [float(value), 1]}
                              for i, value in reversed(list(enumerate(payload["input"]))) ]}, {}, 0
    remote["respond"] = respond
    client = ModelClient(cache_path=tmp_path / "batches.sqlite")
    texts = [str(i) for i in range(130)]
    assert client.embed(texts) == [[float(i), 1] for i in range(130)]
    assert [len(call[1]["input"]) for call in remote["calls"]] == [64, 64, 2]
    assert client.embed(texts) == [[float(i), 1] for i in range(130)]
    assert len(remote["calls"]) == 3


def test_embedding_batches_obey_total_character_budget(remote, monkeypatch):
    monkeypatch.setenv("MODEL_EMBEDDING_BATCH_SIZE", "3")
    monkeypatch.setenv("MODEL_EMBEDDING_BATCH_CHARACTERS", "8")
    client = ModelClient()
    assert len(client.embed(["aaaa", "bbbb", "cc", "dd", "e"])) == 5
    assert [call[1]["input"] for call in remote["calls"]] == [["aaaa", "bbbb"], ["cc", "dd", "e"]]


def test_embedding_rejects_oversize_single_text_before_any_request(remote, monkeypatch):
    monkeypatch.setenv("MODEL_EMBEDDING_BATCH_CHARACTERS", "8")
    with pytest.raises(ModelError, match="character budget"):
        ModelClient().embed(["ok", "longer than budget"])
    assert remote["calls"] == []


def test_chat_markdown_instead_of_json_is_rejected_without_retry(remote):
    remote["respond"] = lambda *_: (200, {"choices": [{"message": {"content": '```json\n{"low_level":[],"high_level":[]}\n```'}}]}, {}, 0)
    with pytest.raises(ModelError, match="JSON"):
        ModelClient().keywords("Pod")
    assert len(remote["calls"]) == 1


def test_remote_answer_can_explicitly_decline_insufficient_evidence(remote):
    remote["respond"] = lambda *_: (200, {"choices": [{"message": {"content": "INSUFFICIENT_EVIDENCE"}}]}, {}, 0)
    assert ModelClient().answer("Missing detail?", [{"text": "Pod runs.", "citation_id": "S1"}], []) == "INSUFFICIENT_EVIDENCE"


def test_chat_receives_only_evidence_fields_without_vectors_or_internal_metadata(remote):
    client = ModelClient()
    contexts = [{"text": "Pod uses Container.", "citation_id": "S1", "embedding": [1, 2], "internal": "private-context-data"}]
    relations = [{"source": "Pod", "target": "Container", "relation": "uses", "description": "Pod uses Container.",
                  "evidence": "Pod uses Container.", "keywords": ["uses"], "embedding": [3, 4], "internal": "private-relation-data"}]
    remote["respond"] = lambda *_: (200, {"choices": [{"message": {"content": "Pod uses Container. [S1]"}}]}, {}, 0)
    assert "[S1]" in client.answer("What?", contexts, relations)
    remote["respond"] = lambda *_: chat({"scores": [{"index": 0, "score": 1}]})
    ranked = client.rerank("Pod", contexts)
    assert ranked[0]["embedding"] == [1, 2]
    for call in remote["calls"]:
        prompt = call[1]["messages"][1]["content"]
        assert "embedding" not in prompt
        assert "private-context-data" not in prompt
        assert "private-relation-data" not in prompt


def test_offline_extracts_generic_operational_verbs_across_lines():
    text = ("Alpha shares a Beta.\nGamma supports Delta.\nRunner executes Task.\n"
            "Registry provides Images.\nDNS resolves Service.\nMonitor identifies Failure.\nPlugin implements Protocol.")
    graph = ModelClient(offline=True).extract(text)
    assert {r["relation"] for r in graph["relations"]} == {"shares", "supports", "executes", "provides", "resolves", "identifies", "implements"}
    assert all(r["evidence"] in text for r in graph["relations"])


def test_offline_preserves_evidence_in_line_wrapped_natural_sentences():
    text = "Service routes to\nPod.\nProbe\nchecks Health."
    graph = ModelClient(offline=True).extract(text)
    assert [(r["source"], r["target"], r["evidence"]) for r in graph["relations"]] == [
        ("Service", "Pod", "Service routes to\nPod."), ("Probe", "Health", "Probe\nchecks Health.")]


def test_answer_wire_contract_is_personal_cross_file_and_preserves_conflicting_evidence(remote):
    contexts = [{"citation_id": "S1", "text": "Project delivery is Monday."},
                {"citation_id": "S2", "text": "Project delivery is Friday."}]
    remote["respond"] = lambda *_: (200, {"choices": [{"message": {"content": "The documents conflict: Monday [S1] versus Friday [S2]."}}]}, {}, 0)
    answer = ModelClient().answer("When is project delivery?", contexts, [])
    assert "[S1]" in answer and "[S2]" in answer
    request = remote["calls"][0][1]
    instruction = request["messages"][0]["content"].lower()
    assert "personal knowledge" in instruction and "multiple documents" in instruction
    assert "conflict" in instruction and "both" in instruction
    assert "do not choose" in instruction
    assert "technical support" not in instruction
    assert json.loads(request["messages"][1]["content"])["data"]["contexts"] == contexts


def test_answer_maps_stored_relation_endpoints_and_keeps_source_citation(remote):
    relations = [{"source_key": "research notes", "target_key": "experiment alpha", "relation": "tracks",
                  "description": "Research notes tracks Experiment Alpha.",
                  "evidence": "Research notes tracks Experiment Alpha.", "citation_id": "S1",
                  "embedding": [0.1], "internal_row_id": 31}]
    remote["respond"] = lambda *_: (200, {"choices": [{"message": {"content": "Research notes tracks Experiment Alpha. [S1]"}}]}, {}, 0)
    ModelClient().answer("What is tracked?", [{"citation_id": "S1", "text": "Research notes tracks Experiment Alpha."}], relations)
    data = json.loads(remote["calls"][0][1]["messages"][1]["content"])["data"]
    assert data["relations"] == [{"source": "research notes", "target": "experiment alpha", "relation": "tracks",
                                  "description": "Research notes tracks Experiment Alpha.",
                                  "evidence": "Research notes tracks Experiment Alpha.", "citation_id": "S1"}]
    assert "source" not in relations[0]


def test_offline_tracks_personal_entities_with_original_evidence():
    text = "Notebook tracks Reading List."
    assert ModelClient(offline=True).extract(text)["relations"] == [
        {"source": "Notebook", "target": "Reading List", "relation": "tracks", "description": text,
         "keywords": ["tracks"], "evidence": text}]
