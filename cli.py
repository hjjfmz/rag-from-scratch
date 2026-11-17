"""Personal local knowledge CLI. Web and CLI share one service stack (GraphService)."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


class CLIParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def _common(parser, child=False):
    default = argparse.SUPPRESS if child else None
    parser.add_argument("--data-dir", default=default, help="SQLite/cache directory (default: project .runtime/knowledge or .runtime/offline)")
    parser.add_argument("--offline", action="store_true", default=argparse.SUPPRESS if child else False,
                        help="Explicit heuristic extraction and lexical vectors; no model service")
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS if child else False, help="Machine-readable stdout")


def build_parser():
    from rag.graph_retrieval import MODES
    parser = CLIParser(description="Local Knowledge — cross-file questions with local/global graph evidence")
    _common(parser)
    commands = parser.add_subparsers(dest="command", required=True, parser_class=CLIParser)
    def command(name, help):
        child = commands.add_parser(name, help=help)
        _common(child, child=True)
        return child
    def retrieval(child):
        child.add_argument("--search-mode", "-m", choices=MODES, default="hybrid_graph")
        child.add_argument("--top-k", "-k", type=int, default=5)
        child.add_argument("--threshold", "-t", type=float, default=.25)
        child.add_argument("--rerank", action="store_true")
        child.add_argument("--verbose", "-v", action="store_true")
    ingest = command("ingest", "Atomically index/replace a document, including triples")
    ingest.add_argument("--source", "-s", required=True)
    ingest.add_argument("--chunk-size", type=int, default=900)
    ingest.add_argument("--overlap", type=int, default=120)
    ingest.add_argument("--workers", type=int, default=4)
    ingest.add_argument("--force", action="store_true", help="Rebuild graph and embeddings for this source")
    sync = command("sync", "Content-aware directory sync")
    sync.add_argument("--source", "-s", required=True)
    sync.add_argument("--force", action="store_true")
    sync.add_argument("--prune", action="store_true", help="Delete indexed local sources missing from this directory")
    for name in ("query", "search", "agentic-query"):
        child = command(name, "Grounded answer" if name == "query" else "Inspect evidence" if name == "search" else "Bounded multi-query graph agent")
        child.add_argument("question")
        retrieval(child)
        if name == "agentic-query":
            child.add_argument("--max-iterations", "-i", type=int, default=3)
    command("list", "List indexed sources and revisions")
    delete = command("delete", "Delete a source and its graph/vector evidence")
    delete.add_argument("--source", "-s", required=True)
    graph = command("graph", "Inspect/export evidence triples")
    graph.add_argument("--output", help="Optional JSON export file")
    doctor = command("doctor", "Check storage, configuration and model identity")
    doctor.add_argument("--live", action="store_true", help="Actually call embedding/chat endpoints (may incur cost)")
    evaluation = command("eval", "Compare source retrieval, graph evidence and latency")
    evaluation.add_argument("--test-file", required=True, help="Your independently labeled JSON question file")
    evaluation.add_argument("--modes", nargs="+", choices=MODES, default=["hybrid", "local", "global", "hybrid_graph"])
    evaluation.add_argument("--search-mode", "-m", choices=MODES, help="Evaluate just this mode")
    evaluation.add_argument("--top-k", "-k", type=int, default=5)
    evaluation.add_argument("--threshold", "-t", type=float, default=.25)
    evaluation.add_argument("--output")
    evaluation.add_argument("--compare", action="store_true", help="Compare hybrid and hybrid_graph")
    return parser


def _write_json(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)
    return str(path)


def _questions(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Evaluation file must be a JSON question list")
    return value


def execute(args):
    from rag.graph_service import GraphService
    from rag.model_client import ModelClient, ModelError
    from rag.settings import load_settings, build_store
    offline = args.offline or os.getenv("RAG_OFFLINE", "").strip().lower() in ("1", "true", "yes")
    settings = load_settings()
    if args.data_dir:
        # Explicit --data-dir pins the local SQLite adapter, overriding RAG_STORAGE.
        settings.storage = "local"
        settings.data_dir = Path(args.data_dir).resolve()
    data_dir = settings.data_dir
    client = None
    if args.command not in ("list", "delete", "graph"):
        try:
            client = ModelClient(offline=offline, cache_path=data_dir / "model-cache.sqlite3")
        except ModelError:
            if args.command != "doctor":
                raise
    store = build_store(settings) if settings.storage == "services" else None
    qdrant = None
    if settings.storage == "services" and client is not None:
        from rag.qdrant_store import QdrantStore
        qdrant = QdrantStore(settings.qdrant_url, client.fingerprint)
    cache = None
    if settings.redis_url:
        from rag.cache import AnswerCache
        cache = AnswerCache(settings.redis_url)
    service = GraphService(data_dir if settings.storage == "local" else None, client,
                           store=store, qdrant=qdrant, cache=cache,
                           chunk_size=getattr(args, "chunk_size", 900),
                           overlap=getattr(args, "overlap", 120), workers=getattr(args, "workers", 4))
    if args.command == "ingest":
        return service.ingest(args.source, force=args.force), 0
    if args.command == "sync":
        result = service.sync(args.source, prune=args.prune, force=args.force)
        return result, int(bool(result["errors"]))
    if args.command in ("query", "search", "agentic-query"):
        options = {"mode": args.search_mode, "top_k": args.top_k, "threshold": args.threshold, "rerank": args.rerank}
        if args.command == "agentic-query":
            result = service.agentic_query(args.question, max_iterations=args.max_iterations, **options)
        else:
            result = getattr(service, args.command)(args.question, **options)
        return result, 0
    if args.command == "list":
        return {"sources": service.list_sources(), "data_dir": str(data_dir)}, 0
    if args.command == "delete":
        return service.delete(args.source), 0
    if args.command == "graph":
        result = service.graph()
        if args.output:
            result["export_path"] = _write_json(args.output, result)
        return result, 0
    if args.command == "doctor":
        result = service.doctor()
        if args.live:
            if client is None:
                raise ValueError("Configure a model provider before --live verification")
            client.embed(["Model endpoint connectivity check"])
            client.extract("Service routes to Pod.")
            result["provider"] = {**client.health(), "network_verified": not offline}
        return result, int(not (result["storage"]["ok"] and result["provider"]["configured"] and result["model_compatible"]))
    if args.command == "eval":
        modes = [args.search_mode] if args.search_mode else (["hybrid", "hybrid_graph"] if args.compare else args.modes)
        result = service.evaluate(_questions(args.test_file), modes=modes, top_k=args.top_k, threshold=args.threshold)
        output = args.output or data_dir / "evaluation.json"
        result["output_path"] = _write_json(output, result)
        return result, 0
    raise ValueError("Unknown command")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    load_dotenv(Path(os.getenv("RAG_ENV_FILE", str(ROOT / ".env"))), override=False)
    json_output = "--json" in argv
    try:
        args = build_parser().parse_args(argv)
        # Existing parsers print progress; JSON stdout is exclusively the final result.
        with redirect_stdout(sys.stderr):
            result, exit_code = execute(args)
        if json_output:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        elif "answer" in result:
            print(result["answer"])
            for context in result["chunks_used"]:
                print(f"[{context['citation_id']}] {context['metadata']['source']}")
            if getattr(args, "verbose", False):
                print(json.dumps(result["retrieval_trace"], ensure_ascii=False, indent=2))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return exit_code
    except KeyboardInterrupt:
        message, exit_code = "Interrupted; already committed sources remain intact", 130
    except Exception as exc:
        message, exit_code = str(exc), 1
    if json_output:
        print(json.dumps({"status": "error", "error": message}, ensure_ascii=False))
    else:
        print(f"Error: {message}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
