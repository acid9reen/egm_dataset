import argparse
import asyncio
import concurrent.futures
import csv
import json
import statistics
import typing as tp
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
from egmlib import preprocess
from tqdm import tqdm

from egm_dataset.cli.stage_1 import STAGE1_FIELDNAMES, Stage1Schema
from egm_dataset.infra import typing as edtp


class Stage2Namespace(argparse.Namespace):
    dataset_filepath: Path
    target_frequency: edtp.Hz
    num_workers: int


def parse_args() -> Stage2Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--dataset-filepath",
        type=Path,
        help="Path to stage 1 dataset .csv file",
    )

    parser.add_argument(
        "-t",
        "--target-frequency",
        type=int,
        default=5000,
        help="Target signal frequency (in Hz)",
    )

    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=1,
        help="Number of processes",
    )

    return parser.parse_args(namespace=Stage2Namespace())


def read_stage_1_data(path: Path) -> tp.Generator[Stage1Schema, None, None]:
    with open(path, "r") as in_:
        csv_reader = csv.DictReader(in_, fieldnames=STAGE1_FIELDNAMES)

        try:
            _ = next(csv_reader)  # Skip header
        except StopIteration as e:
            raise ValueError("The f#ck is going on here?? Why dataset file is empty?") from e

        row: Stage1Schema
        for row in csv_reader:  # type: ignore
            yield row


def read_signal(path: Path) -> npt.NDArray:
    return np.load(path, mmap_mode="r", allow_pickle=True)


def read_label(path: Path) -> list[list[int]]:
    with open(path, "r") as in_:
        possibly_floating = json.load(in_)

    result: list[list[int]] = []

    for channel in possibly_floating:
        rounded = list(map(round, channel))
        result.append(rounded)

    return result


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
        with open(self._label_output_path / filename, "w") as out:
            json.dump(label, out)


class Stage2Schema(tp.TypedDict):
    signal: str
    label: str
    frequency_hz: edtp.Hz

    # Label related
    num_peaks: int
    mean_peaks_distance: float
    min_peaks_distance: float
    max_peaks_distance: float
    median_peaks_distance: float

    # Signal realted
    signal_max: float
    signal_min: float
    signal_mean: float
    signal_median: float
    length: float


STAGE2_FIELDNAMES = sorted(list(Stage2Schema.__required_keys__))


def calculate_mean_peak_distance(label: list[int]) -> float:
    distances = [b - a for a, b in zip(label, label[1:], strict=False)]

    if not distances:
        return 0

    return statistics.mean(distances)


def calculate_median_peak_distance(label: list[int]) -> float:
    distances = [b - a for a, b in zip(label, label[1:], strict=False)]

    if not distances:
        return 0

    return statistics.median(distances)


def calculate_min_peak_distance(label: list[int]) -> float:
    distances = [b - a for a, b in zip(label, label[1:], strict=False)]

    if not distances:
        return 0

    return min(distances)


def calculate_max_peak_distance(label: list[int]) -> float:
    distances = [b - a for a, b in zip(label, label[1:], strict=False)]

    if not distances:
        return 0

    return max(distances)


class Processor:
    def __init__(self, saver: Saver, target_frequency: edtp.Hz, base_path: Path) -> None:
        self._saver = saver
        self._target_frequency = target_frequency
        self._base_path = base_path

    def process_row(self, row: Stage1Schema) -> list[Stage2Schema]:
        signal_path = self._base_path / row["signal"]
        label_path = self._base_path / row["label"]
        frequency = int(row["frequency_hz"])
        q, r = divmod(frequency, self._target_frequency)

        if r != 0:
            raise ValueError("Mod (frequency, target_frequance) should be equal 0")

        signal = read_signal(signal_path)
        label = read_label(label_path)

        result: list[Stage2Schema] = []

        for channel_no, (signal_channel, label_channel) in enumerate(
            zip(signal, label, strict=True),
        ):
            preprocessed_signal_channel: npt.NDArray = preprocess.downsample(
                signal_channel,
                frequency,
                target_frequency=self._target_frequency,
            )

            preprocessed_label_channel = sorted(list(map(lambda x: round(x / q), label_channel)))

            signal_name = "{}_ch_{}{}".format(signal_path.stem, channel_no, signal_path.suffix)
            self._saver.save_signal(preprocessed_signal_channel, signal_name)

            label_name = "{}_ch_{}{}".format(label_path.stem, channel_no, label_path.suffix)
            self._saver.save_label(preprocessed_label_channel, label_name)

            result.append(
                Stage2Schema(
                    signal=(self._saver._signal_output_path / signal_name)
                    .relative_to(self._saver._output_path)
                    .as_posix(),
                    label=(self._saver._label_output_path / label_name)
                    .relative_to(self._saver._output_path)
                    .as_posix(),
                    frequency_hz=self._target_frequency,
                    num_peaks=len(preprocessed_label_channel),
                    mean_peaks_distance=calculate_mean_peak_distance(preprocessed_label_channel),
                    min_peaks_distance=calculate_min_peak_distance(preprocessed_label_channel),
                    max_peaks_distance=calculate_max_peak_distance(preprocessed_label_channel),
                    median_peaks_distance=calculate_median_peak_distance(
                        preprocessed_label_channel,
                    ),
                    signal_min=preprocessed_signal_channel.min(),
                    signal_max=preprocessed_signal_channel.max(),
                    signal_mean=preprocessed_signal_channel.mean(),
                    signal_median=float(np.median(preprocessed_signal_channel)),
                    length=len(preprocessed_signal_channel),
                ),
            )

        return result


async def main() -> int:
    args = parse_args()

    output_folder = args.dataset_filepath.resolve().parent.parent / "stage2"

    saver = Saver(output_folder)
    processor = Processor(saver, args.target_frequency, args.dataset_filepath.parent)

    result: list[Stage2Schema] = []
    loop = asyncio.get_running_loop()
    with concurrent.futures.ProcessPoolExecutor(args.num_workers) as pool:
        tasks = []
        for row in read_stage_1_data(args.dataset_filepath):
            tasks.append(loop.run_in_executor(pool, partial(processor.process_row, row)))

        for row_result in tqdm(asyncio.as_completed(tasks), total=len(tasks), colour="green"):
            result.extend(await row_result)

    with open(output_folder / "dataset.csv", "w", newline="") as out:
        csv_writer = csv.DictWriter(out, fieldnames=STAGE2_FIELDNAMES)
        csv_writer.writeheader()
        csv_writer.writerows(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
