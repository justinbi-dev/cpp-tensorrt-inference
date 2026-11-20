"""Benchmark every RF-DETR engine size in FP16 and FP32 with orin_benchmark.

Runs the engines one at a time, keeps each run's per-frame CSV, and writes a summary CSV and a
markdown table to benchmark/results/<timestamp>/. Engines that haven't been built are skipped.
Uses only the Python standard library.

Example:
    python3 benchmark/benchmark.py --frames 500 --warmup 50 --cooldown 15
"""

import argparse
import csv
import datetime
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SIZES = ["nano", "small", "medium", "large"]
PRECISIONS = ["fp16", "fp32"]

STAGE_COLUMNS = [
    ("preprocess_ms", "Preprocess"),
    ("gpu_h2d_ms", "H2D"),
    ("gpu_compute_ms", "TensorRT"),
    ("gpu_d2h_ms", "D2H"),
    ("postprocess_ms", "Postprocess"),
]


def run_command(command: list[str]) -> str:
    """Return a command's output, or a note if it can't be run (e.g. it needs sudo)."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable ({e})"
    output = (result.stdout + result.stderr).strip().replace("\n", " ")
    return output if result.returncode == 0 else f"unavailable ({output})"


def write_environment(path: Path, args: argparse.Namespace) -> None:
    model = Path("/proc/device-tree/model")
    lines = [
        f"date: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"device: {model.read_text().strip(chr(0)) if model.exists() else 'unknown'}",
        f"l4t: {run_command(['head', '-n', '1', '/etc/nv_tegra_release'])}",
        f"tensorrt: {run_command(['dpkg-query', '-W', '-f=${Version}', 'libnvinfer-bin'])}",
        f"power mode: {run_command(['nvpmodel', '-q'])}",
        "jetson_clocks: check by hand, reading it needs sudo (sudo jetson_clocks --show)",
        f"frames: {args.frames}",
        f"warmup: {args.warmup}",
        f"cooldown_s: {args.cooldown}",
    ]
    path.write_text("\n".join(lines) + "\n")


def run_engine(binary: Path, base_config: Path, engine: Path, out_csv: Path,
               frames: int, warmup: int) -> None:
    # Relative paths in the config resolve against the config's own directory, so the temporary
    # config goes next to the base config rather than in /tmp.
    text = base_config.read_text()
    lines = text.splitlines()
    replaced = [f'engine_path: "{engine}"' if l.startswith("engine_path:") else l for l in lines]
    if replaced == lines:
        raise RuntimeError(f"No engine_path line found in {base_config}")

    fd, tmp_name = tempfile.mkstemp(suffix=".yaml", prefix=".bench_", dir=base_config.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(replaced) + "\n")
        subprocess.run([str(binary), tmp_name, str(frames), str(warmup), str(out_csv)],
                       check=True)
    finally:
        os.remove(tmp_name)


def summarize_run(csv_path: Path) -> dict:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        raise RuntimeError(f"{csv_path} has fewer than 2 frames, too few to summarize")

    def column(name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    total = column("total_ms")
    percentiles = statistics.quantiles(total, n=100, method="inclusive")
    summary = {
        "frames": len(rows),
        "fps": 1000.0 / statistics.mean(total),
        "mean_ms": statistics.mean(total),
        "median_ms": statistics.median(total),
        "p95_ms": percentiles[94],
        "p99_ms": percentiles[98],
        "detections_per_frame": statistics.mean(column("num_detections")),
    }
    for name, _ in STAGE_COLUMNS:
        summary[f"median_{name}"] = statistics.median(column(name))
        stage_percentiles = statistics.quantiles(column(name), n=100, method="inclusive")
        summary[f"p95_{name}"] = stage_percentiles[94]
        summary[f"p99_{name}"] = stage_percentiles[98]
    return summary


def write_summary_csv(path: Path, results: list[tuple[str, str, dict]]) -> None:
    fieldnames = ["size", "precision"] + list(results[0][2].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for size, precision, summary in results:
            writer.writerow({"size": size, "precision": precision,
                             **{k: round(v, 3) for k, v in summary.items()}})


def markdown_tables(results: list[tuple[str, str, dict]]) -> str:
    out = [
        "| Model | Precision | FPS | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | Detections/frame |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for size, precision, s in results:
        out.append(f"| {size} | {precision.upper()} | {s['fps']:.1f} | {s['mean_ms']:.2f} | "
                   f"{s['median_ms']:.2f} | {s['p95_ms']:.2f} | {s['p99_ms']:.2f} | "
                   f"{s['detections_per_frame']:.1f} |")

    out += ["", "Median per-stage time (ms):", ""]
    out.append("| Model | Precision | " + " | ".join(label for _, label in STAGE_COLUMNS) + " |")
    out.append("|---|---|" + "---|" * len(STAGE_COLUMNS))
    for size, precision, s in results:
        stages = " | ".join(f"{s[f'median_{name}']:.2f}" for name, _ in STAGE_COLUMNS)
        out.append(f"| {size} | {precision.upper()} | {stages} |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=500, help="Measured frames per engine")
    parser.add_argument("--warmup", type=int, default=50, help="Warm-up frames per engine")
    parser.add_argument("--cooldown", type=float, default=15.0, help="Seconds between runs")
    parser.add_argument("--binary", type=Path, default=REPO_ROOT / "build" / "orin_benchmark")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "default.yaml")
    parser.add_argument("--engine-dir", type=Path, default=REPO_ROOT / "models" / "output")
    args = parser.parse_args()

    if not args.binary.is_file():
        parser.error(f"{args.binary} not found. Build the project first.")

    out_dir = REPO_ROOT / "benchmark" / "results" / datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_environment(out_dir / "environment.txt", args)

    engines = [(size, precision) for size in SIZES for precision in PRECISIONS]
    available = [(s, p) for s, p in engines
                 if (args.engine_dir / f"rfdetr-{s}-{p}.engine").is_file()]
    for s, p in engines:
        if (s, p) not in available:
            print(f"Skipping rfdetr-{s}-{p}: engine not found in {args.engine_dir}")
    if not available:
        raise SystemExit("No engines found, nothing to benchmark.")

    results = []
    for i, (size, precision) in enumerate(available):
        name = f"rfdetr-{size}-{precision}"
        print(f"\n=== {name} ({i + 1}/{len(available)}) ===")
        raw_csv = raw_dir / f"{name}.csv"
        run_engine(args.binary, args.config, args.engine_dir / f"{name}.engine", raw_csv,
                   args.frames, args.warmup)
        results.append((size, precision, summarize_run(raw_csv)))
        if i < len(available) - 1 and args.cooldown > 0:
            print(f"Cooling down for {args.cooldown:g} s")
            time.sleep(args.cooldown)

    write_summary_csv(out_dir / "summary.csv", results)
    markdown = markdown_tables(results)
    (out_dir / "summary.md").write_text(markdown)
    print(f"\nResults written to {out_dir}\n")
    print(markdown)


if __name__ == "__main__":
    main()
