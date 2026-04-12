"""
pocket-mem show — visual memory explorer entry point.

Usage:
    pocket-mem show
    pocket-mem show --project myapp
    pocket-mem show --topic "People I Know" --since 7d
    pocket-mem show --type entity --search David
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pocket-mem",
        description="pocket-mem memory tools",
    )
    sub = parser.add_subparsers(dest="cmd")

    show = sub.add_parser("show", help="Open visual memory explorer in browser")
    show.add_argument(
        "--project", default=None,
        help="Project name (default: auto-detect from --path)",
    )
    show.add_argument(
        "--path", default="./memory",
        help="Directory containing .db files (default: ./memory)",
    )
    show.add_argument(
        "--topic", action="append", default=None, metavar="TOPIC",
        help="Filter to nodes in this topic (repeatable)",
    )
    show.add_argument(
        "--type", action="append", default=None, metavar="TYPE",
        help="Filter to node_type: entity, memory_chunk, topic, tone (repeatable)",
    )
    show.add_argument(
        "--since", default=None, metavar="DATE",
        help="Show nodes updated since: '7d', '30d', or ISO date (e.g. 2025-01-01)",
    )
    show.add_argument(
        "--search", default=None, metavar="QUERY",
        help="Pre-fill search bar and highlight matching nodes",
    )
    show.add_argument(
        "--port", type=int, default=None,
        help="Port for local server (default: random 8700–8799)",
    )

    args = parser.parse_args()

    if args.cmd == "show":
        from memory_agent.visualizer.query import build_graph
        from memory_agent.visualizer.server import launch

        graph = build_graph(
            path=args.path,
            project=args.project,
            topics=args.topic,
            types=args.type,
            since=args.since,
            search=args.search,
        )
        n = len(graph["nodes"])
        e = len(graph["edges"])
        print(f"  pocket-mem: {n} nodes, {e} edges — {graph['meta']['db_path']}")
        launch(graph, port=args.port, initial_search=args.search)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
