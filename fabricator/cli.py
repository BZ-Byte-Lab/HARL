from __future__ import annotations

import argparse
from pathlib import Path

from fabricator.export import write_csv, write_jsonl
from fabricator.fabricator import Fabricator
from fabricator.personas import PERSONAS


def _write_outputs(rows, persona_name: str, weeks: int, seed: int, out_dir: Path, output_format: str) -> list[Path]:
    base_dir = out_dir / persona_name
    stem = f"weeks{weeks}_seed{seed}"
    written: list[Path] = []
    if output_format in {"csv", "both"}:
        written.append(write_csv(rows, base_dir / f"{stem}.csv"))
    if output_format in {"jsonl", "both"}:
        written.append(write_jsonl(rows, base_dir / f"{stem}.jsonl"))
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate persona-labeled smart-home event datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate rows for one persona.")
    generate.add_argument("--persona", choices=sorted(PERSONAS), required=True)
    generate.add_argument("--weeks", type=int, default=4)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--format", choices=("csv", "jsonl", "both"), default="both")
    generate.add_argument("--out", type=Path, default=Path("artifacts/fabricated"))

    generate_all = subparsers.add_parser("generate-all", help="Generate rows for all personas.")
    generate_all.add_argument("--weeks", type=int, default=4)
    generate_all.add_argument("--seed", type=int, default=42)
    generate_all.add_argument("--format", choices=("csv", "jsonl", "both"), default="both")
    generate_all.add_argument("--out", type=Path, default=Path("artifacts/fabricated"))

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "generate":
        rows = Fabricator(persona=args.persona, seed=args.seed).generate_dataset(args.weeks)
        written = _write_outputs(rows, args.persona, args.weeks, args.seed, args.out, args.format)
        print(f"persona={args.persona}")
        print(f"rows={len(rows)}")
        for path in written:
            print(f"wrote={path}")
        return

    total_rows = 0
    for persona_name in sorted(PERSONAS):
        rows = Fabricator(persona=persona_name, seed=args.seed).generate_dataset(args.weeks)
        written = _write_outputs(rows, persona_name, args.weeks, args.seed, args.out, args.format)
        total_rows += len(rows)
        print(f"persona={persona_name} rows={len(rows)}")
        for path in written:
            print(f"wrote={path}")
    print(f"total_rows={total_rows}")


if __name__ == "__main__":
    main()
