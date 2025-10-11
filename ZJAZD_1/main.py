from __future__ import annotations
import argparse
from typing import List

from tug_of_war import play_human_vs_ai, play_ai_vs_ai


def parse_moves(raw: list[str]) -> List[int]:
    try:
        moves = [int(x) for x in raw]
    except ValueError as e:
        raise SystemExit(f"Invalid --moves values: {raw}. Must be integers.") from e
    if any(m <= 0 for m in moves):
        raise SystemExit(f"Moves must be positive integers, got: {moves}")
    return moves


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Arithmetic Tug-of-War (easyAI)")
    parser.add_argument("--mode", choices=["human_vs_ai", "ai_vs_ai"], default="human_vs_ai",
                        help="Choose gameplay mode.")
    parser.add_argument("--k", type=int, default=5, help="Target boundary K (default 5).")
    parser.add_argument("--moves", nargs="+", default=[1, 2, 3],
                        help="Allowed magnitudes, e.g. --moves 1 2 3 (default).")
    parser.add_argument("--depth", type=int, default=8, help="Negamax search depth (default 8).")
    parser.add_argument("--depth2", type=int, default=None,
                        help="Negamax depth for Player 2 (AI-vs-AI only). Defaults to --depth.")
    args = parser.parse_args(argv)

    if args.k <= 0:
        raise SystemExit("--k must be a positive integer.")

    moves = parse_moves(args.moves)
    depth1 = int(args.depth)
    depth2 = int(args.depth2) if args.depth2 is not None else depth1

    if args.mode == "human_vs_ai":
        play_human_vs_ai(k=args.k, moves=moves, depth=depth1)
    else:
        play_ai_vs_ai(k=args.k, moves=moves, depth1=depth1, depth2=depth2)


if __name__ == "__main__":
    main()