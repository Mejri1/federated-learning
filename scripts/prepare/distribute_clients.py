"""
Split normalized member datasets into client folders with 25% per client.

The script:
- reads data/normalized/<member>/<memberXX>.jpg
- assigns images equally across clients (default: 4 clients)
- copies them into data/clients/<client>/<member>
- saves assignment metadata CSV
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Sequence


DEFAULT_MEMBERS = ["abir", "jihene", "omarbr", "omarmej"]
DEFAULT_CLIENTS = ["client_abir", "client_jihene", "client_omarbr", "client_omarmej"]


def chunk_counts(total: int, buckets: int) -> tuple[list[int], int]:
    base = total // buckets
    counts = [base] * buckets
    remainder = total - base * buckets
    return counts, remainder


def prepare_output_dirs(output_root: Path, clients: Sequence[str], members: Sequence[str], clean: bool) -> None:
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    for client in clients:
        for member in members:
            (output_root / client / member).mkdir(parents=True, exist_ok=True)


def distribute_images(
    members: Sequence[str],
    clients: Sequence[str],
    source_root: Path,
    output_root: Path,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    totals = {client: 0 for client in clients}
    per_member_allocation = {client: {member: 0 for member in members} for client in clients}
    dropped: dict[str, int] = {member: 0 for member in members}

    for member in members:
        member_dir = source_root / member
        if not member_dir.exists():
            raise FileNotFoundError(f"Missing normalized folder for member '{member}': {member_dir}")

        images = sorted(p for p in member_dir.glob("*.jpg") if p.is_file())
        if not images:
            continue

        rng.shuffle(images)
        counts, remainder = chunk_counts(len(images), len(clients))
        if remainder:
            dropped[member] += remainder
        idx = 0

        for client, share in zip(clients, counts):
            target_dir = output_root / client / member
            for _ in range(share):
                src = images[idx]
                dst = target_dir / src.name
                shutil.copy2(src, dst)
                rows.append({"client": client, "member": member, "filename": src.name})
                totals[client] += 1
                per_member_allocation[client][member] += 1
                idx += 1

    return rows, totals, per_member_allocation, dropped


def write_assignment_csv(rows: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["client", "member", "filename"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distribute normalized member data across clients.")
    parser.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS, help="Member folders to include.")
    parser.add_argument("--clients", nargs="+", default=DEFAULT_CLIENTS, help="Client names to create.")
    parser.add_argument("--source", type=Path, default="data/normalized", help="Normalized data root.")
    parser.add_argument("--output", type=Path, default="data/clients", help="Output root for client data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument("--no-clean", action="store_true", help="Do not wipe the output directory before copying.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output_dirs(args.output, args.clients, args.members, clean=not args.no_clean)
    rows, totals, per_member_allocation, dropped = distribute_images(args.members, args.clients, args.source, args.output, args.seed)
    write_assignment_csv(rows, args.output / "client_assignment.csv")

    print("Distribution summary:")
    for client, count in totals.items():
        member_breakdown = ", ".join(f"{member}:{per_member_allocation[client][member]}" for member in args.members)
        print(f"  {client}: {count} images ({member_breakdown})")
    if any(dropped.values()):
        print("Dropped leftovers per member (not divisible by number of clients):")
        for member, amount in dropped.items():
            if amount:
                print(f"  {member}: {amount} images skipped")
    print(f"Metadata: {args.output / 'client_assignment.csv'}")


if __name__ == "__main__":
    main()

