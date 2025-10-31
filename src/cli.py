import argparse
import json
import sys
from pathlib import Path

from src.ingest import ingest_path, ingest_youtube
from src.retriever import ask

def _print_json(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))

def main():
    p = argparse.ArgumentParser(prog="multimodal-rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest a file or folder")
    p_ing.add_argument("path", help="File or directory to ingest")

    p_yti = sub.add_parser("ingest-yt", help="Ingest a YouTube URL (audio transcript)")
    p_yti.add_argument("url", help="YouTube URL")

    p_ask = sub.add_parser("ask", help="Query the knowledge base")
    p_ask.add_argument("question", help="Your question")
    p_ask.add_argument("--top_k", type=int, default=6)
    p_ask.add_argument("--only", choices=["all","audio","video","images","youtube"], default="all")
    p_ask.add_argument("--file", help="Restrict to filename substring")
    p_ask.add_argument("--url_contains", help="Restrict to URL substring")

    args = p.parse_args()

    if args.cmd == "ingest":
        res = ingest_path(args.path)
        _print_json(res)
        return

    if args.cmd == "ingest-yt":
        res = ingest_youtube(args.url)
        _print_json(res)
        return

    if args.cmd == "ask":
        where = None
        if args.only == "youtube":
            where = {"source": "youtube"}
        elif args.only == "images":
            where = {"type": "image"}
        elif args.only == "audio":
            where = {"type": "audio"}
        elif args.only == "video":
            where = {"type": "video"}
        if args.file:
            where = (where or {}) | {"path_contains": args.file}
        if args.url_contains:
            where = (where or {}) | {"url_contains": args.url_contains}

        res = ask(args.question, top_k=args.top_k, where=where)
        _print_json(res)
        return

if __name__ == "__main__":
    main()
