import argparse
import asyncio
import concurrent.futures
import csv
import json
import random
import typing as tp
from bisect import bisect_left
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np
import numpy.typing as npt
from egmlib.preprocess import highpass_filter, moving_avg_filter
from scipy.stats import norm
from tqdm import tqdm

from egm_dataset.cli.stage_2 import STAGE2_FIELDNAMES, Stage2Schema


class Stage3Namespace(argparse.Namespace):
    dataset_filepath: Path
    output_folder: str
    critical_frequency: int
    mean_avg_size: int
    scale_factor: float
    num_workers: int
    signal_frequency: int
    length: int


def parse_args() -> Stage3Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--dataset-filepath",
        type=Path,
        help="Path to stage 2 dataset .csv file",
    )

    parser.add_argument("-o", "--output-folder", help="Output folder name")

    parser.add_argument(
        "-c",
        "--critical-frequency",
        type=int,
        default=100,
        help="Critical frequency for highpass filter",
    )

    parser.add_argument(
        "--signal-frequency",
        type=int,
        default=5000,
        help="Input signal frequency",
    )

    parser.add_argument(
        "--mean-avg-size",
        type=int,
        default=3,
        help="Size of sliding window for mean average filter",
    )

    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=1,
        help="Number of processes",
    )

    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=10_000,
        help="Length of output signal/label segments",
    )

    parser.add_argument(
        "-s",
        "--scale-factor",
        type=float,
        default=1000,
        help="Scale factor for signal scaling",
    )

    return parser.parse_args(namespace=Stage3Namespace())


class Stage3Schema(tp.TypedDict):
    signal: str
    label: str
    experiment: str
    channel: int


STAGE3_FIELDNAMES = sorted(list(Stage3Schema.__required_keys__))


def read_dataset_file(file: Path) -> tp.Generator[Stage2Schema, None, None]:
    with open(file, "r", newline="") as in_:
        csv_reader = csv.DictReader(in_, fieldnames=STAGE2_FIELDNAMES)

        # Skip header
        _ = next(csv_reader)

        row: Stage2Schema
        for row in csv_reader:  # type: ignore
            yield row


class Saver:
    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self._signal_output_path = output_path / "x"
        self._label_output_path = output_path / "y"

        self._setup_folders(self._output_path, self._signal_output_path, self._label_output_path)

    @staticmethod
    def _setup_folders(*folders: Path) -> None:
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)

    def save_signal(self, signal: npt.NDArray, filename: str) -> None:
        np.save(self._signal_output_path / filename, signal)

    def save_label(self, label: tp.Any, filename: str) -> None:
        np.save(self._label_output_path / filename, label)


def read_label(filepath: Path) -> list[int]:
    with open(filepath) as in_:
        label = json.load(in_)

    return label


def read_signal(filepath: Path) -> npt.NDArray[np.float32]:
    signal = np.load(filepath, allow_pickle=True)

    return signal


def gaussian_kernel(sigma: float, size: int) -> npt.NDArray[np.float32]:
    if size < 2:
        raise ValueError("Fuck off")

    rng = range(-size // 2, size // 2 + 1)
    rv = norm(0, sigma)
    kernel = np.array(list(map(rv.pdf, rng)), dtype=np.float32)  # type: ignore
    kernel = kernel / kernel.max()

    return kernel


_T = tp.TypeVar("_T", bound=npt.NDArray)


def preprocess_signal(signal: _T, transforms: tp.Iterable[tp.Callable[[_T], _T]]) -> _T:
    for transform in transforms:
        signal = transform(signal)

    return signal


class Segment(tp.NamedTuple):
    start: int
    stop: int


def find_ge(a, x):
    """Find leftmost item greater than or equal to x"""
    i = bisect_left(a, x)

    if i != len(a):
        return a[i]

    return None


class Processor:
    def __init__(
        self,
        saver: Saver,
        dataset_root: Path,
        signal_transformer: tp.Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]],
        conv_kernel: npt.NDArray,
        length: int,
        max_shift: int,
    ) -> None:
        self._saver = saver
        self._dataset_root = dataset_root
        self._signal_transformer = signal_transformer
        self._conv_kernel = conv_kernel
        self._length = length
        self._max_shift = max_shift

    def _process_single(self, row: Stage2Schema) -> list[Stage3Schema]:
        if row["num_peaks"] == 0:
            return []

        signal_path = self._dataset_root / row["signal"]
        label_path = self._dataset_root / row["label"]

        signal = read_signal(signal_path)
        preprocessed_signal = self._signal_transformer(signal)

        label = read_label(label_path)
        arr = np.zeros(len(signal))

        for pos in label:
            arr[pos] = 1

        preprocessed_label = np.convolve(arr, self._conv_kernel, mode="same")

        result: list[Stage3Schema] = []

        for end_index in range(
            self._length + self._max_shift,
            len(signal),
            self._length + self._max_shift,
        ):
            shift = random.randint(0, self._max_shift - 1)
            end = end_index - shift
            start = end - self._length

            # Skip cutout if there is not labels
            if (found := find_ge(label, start)) is not None:
                if found >= end:
                    continue

            signal_cutout = preprocessed_signal[start:end]
            label_cutout = preprocessed_label[start:end]

            signal_cutout_filename = f"{signal_path.stem}_{start}-{end}{signal_path.suffix}"
            label_cutout_filename = f"{label_path.stem}_{start}-{end}.npy"
            signal_cutout_filepath = self._saver._signal_output_path / signal_cutout_filename
            label_cutout_filepath = self._saver._label_output_path / label_cutout_filename

            schema = Stage3Schema(
                signal=signal_cutout_filepath.relative_to(self._saver._output_path).as_posix(),
                label=label_cutout_filepath.relative_to(self._saver._output_path).as_posix(),
                experiment=signal_cutout_filepath.stem.split("_ch_")[0],
                channel=int(signal_cutout_filepath.stem.split("_ch_")[1].split("_")[0]),
            )

            self._saver.save_signal(signal_cutout, signal_cutout_filename)
            self._saver.save_label(label_cutout, label_cutout_filename)

            result.append(schema)

        return result

    def process(self, rows: tp.Iterable[Stage2Schema]) -> list[Stage3Schema]:
        result: list[Stage3Schema] = []

        for row in rows:
            part = self._process_single(row)
            result.extend(part)

        return result


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def scale(x, factor):
    return x / factor


async def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_filepath.parent
    output_filepath = dataset_root.parent / args.output_folder

    saver = Saver(output_filepath)
    conv_kernel = gaussian_kernel(20, 200)

    conf_highpass_filter = partial(
        highpass_filter,
        frequency=args.signal_frequency,
        order=2,
        critical_frequency=args.critical_frequency,
    )
    conf_moving_avg_filter = partial(moving_avg_filter, size=args.mean_avg_size)

    signal_transformer = partial[npt.NDArray[np.float32]](
        preprocess_signal,
        transforms=[
            conf_highpass_filter,
            conf_moving_avg_filter,
            partial(scale, factor=args.scale_factor),
        ],
    )

    processor = Processor(
        saver,
        dataset_root,
        signal_transformer=signal_transformer,  # type: ignore
        conv_kernel=conv_kernel,
        length=args.length,
        max_shift=2000,
    )

    result: list[Stage3Schema] = []
    loop = asyncio.get_running_loop()

    with concurrent.futures.ProcessPoolExecutor(args.num_workers) as pool:
        tasks = []
        for rows in batched(read_dataset_file(args.dataset_filepath), 60):
            tasks.append(loop.run_in_executor(pool, partial(processor.process, rows)))

        for rows_result in tqdm(asyncio.as_completed(tasks), total=len(tasks), colour="green"):
            result.extend(await rows_result)

    with open(output_filepath / "dataset.csv", "w", newline="") as out:
        csv_writer = csv.DictWriter(out, fieldnames=STAGE3_FIELDNAMES)
        csv_writer.writeheader()
        csv_writer.writerows(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
