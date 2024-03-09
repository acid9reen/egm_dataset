import argparse
import csv
import typing as tp
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


class Stage1Schema(tp.TypedDict):
    signal: str
    label: str
    frequency_hz: int


STAGE1_FIELDNAMES = sorted(list(Stage1Schema.__required_keys__))


def main() -> int:
    args = parse_args()

    xs = sorted(
        map(
            lambda p: p.resolve().relative_to(args.dataset_root).as_posix(),
            args.dataset_root.rglob("*.npy"),
        ),
    )

    ys = sorted(
        map(
            lambda p: p.resolve().relative_to(args.dataset_root).as_posix(),
            args.dataset_root.rglob("*.json"),
        ),
    )

    output_filepath = args.dataset_root / args.output_filename

    existing_xs: set[str] = set()
    existing_ys: set[str] = set()
    if file_already_exists := output_filepath.exists():
        with open(output_filepath, "r") as in_:
            csv_reader = csv.DictReader(in_, fieldnames=STAGE1_FIELDNAMES)
            _ = next(csv_reader)  # Skip header

            row: Stage1Schema
            for row in csv_reader:  # type: ignore
                existing_xs.add(row["signal"])
                existing_ys.add(row["label"])

    with open(output_filepath, "a", newline="") as out:
        csv_writer = csv.DictWriter(out, fieldnames=STAGE1_FIELDNAMES)

        _ = file_already_exists or csv_writer.writeheader()  # Used only for side effect, thus _
        for x, y, frequency in zip(xs, ys, repeat(args.default_frequency)):
            if x in existing_xs and y in existing_ys:
                continue

            csv_writer.writerow(Stage1Schema(signal=x, label=y, frequency_hz=frequency))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
