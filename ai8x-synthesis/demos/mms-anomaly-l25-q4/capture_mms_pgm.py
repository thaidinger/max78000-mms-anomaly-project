#!/usr/bin/env python3
"""Capture MMS PPM/PGM images from the MAX78000 UART output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def read_lines(port: str | None, baud: int):
    if port is None or port == "-":
        yield from sys.stdin
        return

    try:
        import serial  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("pyserial is required for --port. Install with: pip install pyserial") from exc

    with serial.Serial(port, baudrate=baud, timeout=None) as ser:
        while True:
            yield ser.readline().decode("ascii", errors="ignore")


def collect_block(lines, begin: str, end: str) -> list[str]:
    in_block = False
    block: list[str] = []

    for line in lines:
        text = line.strip()
        if text == begin:
            in_block = True
            block = []
            continue
        if not in_block and text:
            print(f"UART: {text}")
        if text == end and in_block:
            return block
        if in_block:
            block.append(line)

    raise SystemExit(f"Did not receive {begin} ... {end}")


def extract_block(lines: list[str], begin: str, end: str) -> list[str]:
    in_block = False
    block: list[str] = []

    for line in lines:
        text = line.strip()
        if text == begin:
            in_block = True
            block = []
            continue
        if text == end and in_block:
            return block
        if in_block:
            block.append(line)

    raise SystemExit(f"Frame did not contain {begin} ... {end}")


def parse_info(lines: list[str]) -> dict[str, str]:
    info: dict[str, str] = {}

    for line in lines:
        if "=" not in line:
            continue
        key, value = line.strip().split("=", 1)
        info[key] = value

    return info


def print_info(info: dict[str, str], index: int) -> None:
    source = info.get("source", "UNKNOWN")
    guess = info.get("guess", "UNKNOWN")
    true = info.get("true", "UNKNOWN")
    score = int(info.get("image_score_q1000", "0"))
    threshold = int(info.get("threshold_q1000", "0"))
    anomalous = info.get("anomalous_cells", "?")
    cells = info.get("num_cells", "?")
    max_margin = info.get("max_margin", "?")
    mean_margin = info.get("mean_margin", "?")

    print(f"\nFrame {index:03d}")
    print(f"  Source: {source}")
    print(f"  Guess: {guess}")
    print(f"  True: {true}")
    print(f"  Image anomaly score: {score / 1000:.3f}, threshold: {threshold / 1000:.3f}")
    print(f"  Anomalous cells: {anomalous}/{cells}")
    print(f"  Max anomaly margin: {max_margin}")
    print(f"  Mean anomaly margin: {mean_margin}")


def write_preview(input_ppm: Path, anomaly_pgm: Path, preview_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        print("matplotlib not installed; wrote PPM/PGM files only.")
        return

    def load_pnm(path: Path):
        with path.open("r", encoding="ascii") as f:
            magic = f.readline().strip()
            if magic not in ("P2", "P3"):
                raise ValueError(f"{path} is not an ASCII PGM/PPM")
            line = f.readline().strip()
            while line.startswith("#"):
                line = f.readline().strip()
            width, height = map(int, line.split())
            maxval = int(f.readline().strip())
            values = [int(v) for v in f.read().split()]

        channels = 3 if magic == "P3" else 1
        expected = width * height * channels
        if len(values) != expected:
            raise ValueError(f"{path} has {len(values)} values, expected {expected}")
        return width, height, maxval, channels, values

    in_w, in_h, _, in_channels, in_vals = load_pnm(input_ppm)
    map_w, map_h, _, _, map_vals = load_pnm(anomaly_pgm)
    if in_channels == 3:
        input_img = [
            [in_vals[3 * (row * in_w + col):3 * (row * in_w + col) + 3]
             for col in range(in_w)]
            for row in range(in_h)
        ]
    else:
        input_img = [in_vals[row * in_w:(row + 1) * in_w] for row in range(in_h)]
    anomaly_img = [map_vals[row * map_w:(row + 1) * map_w] for row in range(map_h)]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(input_img, vmin=0, vmax=255)
    axes[0].set_title("Input")
    axes[1].imshow(anomaly_img, cmap="inferno", vmin=0, vmax=255)
    axes[1].set_title("Anomaly map")
    axes[2].imshow(input_img, vmin=0, vmax=255)
    axes[2].imshow(anomaly_img, cmap="inferno", vmin=0, vmax=255, alpha=0.45,
                   extent=(0, in_w, in_h, 0), interpolation="nearest")
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(preview_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {preview_png}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="UART port, for example /dev/cu.usbmodemXXXX. Use '-' for stdin.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--out-dir", default="uart_frames")
    parser.add_argument("--prefix", default="mms_frame")
    parser.add_argument("--count", type=int, default=0,
                        help="Number of frames to capture. 0 means run until Ctrl-C.")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = read_lines(args.port, args.baud)
    print("Waiting for BEGIN_MMS_FRAME...")
    index = 0

    while args.count == 0 or index < args.count:
        frame = collect_block(lines, "BEGIN_MMS_FRAME", "END_MMS_FRAME")
        info_block = extract_block(frame, "BEGIN_MMS_INFO", "END_MMS_INFO")
        input_block = extract_block(frame, "BEGIN_INPUT_PPM", "END_INPUT_PPM")
        anomaly_block = extract_block(frame, "BEGIN_ANOMALY_PGM", "END_ANOMALY_PGM")
        frame_prefix = f"{args.prefix}_{index:03d}"

        print_info(parse_info(info_block), index)

        input_ppm = out_dir / f"{frame_prefix}_input.ppm"
        anomaly_pgm = out_dir / f"{frame_prefix}_anomaly.pgm"
        input_ppm.write_text("".join(input_block), encoding="ascii")
        anomaly_pgm.write_text("".join(anomaly_block), encoding="ascii")
        print(f"  Wrote {input_ppm}")
        print(f"  Wrote {anomaly_pgm}")

        if not args.no_preview:
            write_preview(input_ppm, anomaly_pgm, out_dir / f"{frame_prefix}_preview.png")

        index += 1
        print("Waiting for next BEGIN_MMS_FRAME...")


if __name__ == "__main__":
    main()
