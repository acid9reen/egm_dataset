import argparse
import csv
from itertools import repeat
from pathlib import Path

from egm_dataset.infra import typing as edtp


class Stage1Namespace(argparse.Namespace):
    dataset_root: Path
    default_frequency: int
    output_filename: str


def parse_args() -> Stage1Namespace:
    parser = argparse.ArgumentParser(
        "EGM Dataset Stage 1",
        description="""
            Perform raw data preparation,
            collect all data and labes,
            also fill frequency field with default
        """,
    )

    parser.add_argument(
        "-r",
        "--dataset_root",
        type=Path,
        help="Path to dataset root (folder with .npy and .json)",
    )

    parser.add_argument(
        "-f",
        "--default_frequency",
        type=edtp.Hz,
        help="Default frequency to fill new signals `frequecy` field",
        default=20_000,
    )

    parser.add_argument(
        "-o",
        "--output_filename",
        help="Output filename (name of the produced .csv file with extension)",
        default="dataset.csv",
    )

    return parser.parse_args(namespace=Stage1Namespace())


def main() -> int:
    args = parse_args()

    xs = sorted(
        map(
            lambda p: p.resolve().relative_to(args.dataset_root),
            args.dataset_root.rglob("*.npy"),
        ),
    )

    ys = sorted(
        map(
            lambda p: p.resolve().relative_to(args.dataset_root),
            args.dataset_root.rglob("*.json"),
        ),
    )

    with open(args.dataset_root / args.output_filename, "w", newline="") as out:
        csv_writer = csv.writer(out)

        csv_writer.writerow(["signal", "label", "frequency(hz)"])
        csv_writer.writerows(zip(xs, ys, repeat(args.default_frequency)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
