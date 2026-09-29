from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 패키지를 설치하지 않고 이 파일을 직접 실행해도 src의 모듈을 찾게 한다.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepvoice_diffusion.audio import (  # noqa: E402
    load_waveform,
    make_training_segment,
    segment_to_logmel,
)
from deepvoice_diffusion.config import load_config  # noqa: E402
from deepvoice_diffusion.dataset import find_wav_files, select_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    """명령행에서 설정 파일, 임시 처리 개수, 출력 위치를 받을 수 있게 한다."""
    parser = argparse.ArgumentParser(description="Convert WAV files to normalized log-Mels.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocess.yaml",
    )
    parser.add_argument("--limit", type=int, default=None, help="Temporary smoke-test limit.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def output_name(index: int, path: Path, input_dir: Path) -> str:
    """긴 한글 원본명 대신 순번과 경로 hash로 안정적인 출력 파일명을 만든다."""
    relative = path.relative_to(input_dir).as_posix()
    # 같은 상대 경로는 항상 같은 10자리 식별자를 만든다.
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
    return f"{index:05d}_{digest}.npy"


def random_generator_for_file(
    path: Path,
    input_dir: Path,
    base_seed: int,
) -> np.random.Generator:
    """파일별로 독립적이면서 재현 가능한 난수 생성기를 만든다."""
    relative = path.relative_to(input_dir).as_posix()
    # 공통 seed와 파일 경로를 함께 hash한다. 파일 목록의 순서가 바뀌어도
    # 같은 파일은 같은 random crop 시작점을 갖는다.
    seed_material = f"{base_seed}:{relative}".encode("utf-8")
    file_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    return np.random.default_rng(file_seed)


def main() -> None:
    # 1) YAML 설정을 읽고 입력 경로와 파라미터가 유효한지 검사한다.
    args = parse_args()
    config = load_config(args.config)

    # 2) WAV를 재귀적으로 찾고 최소 개수 조건을 먼저 확인한다.
    files = select_files(
        find_wav_files(config.dataset.input_dir),
        config.dataset.min_samples,
        config.dataset.max_samples,
    )
    if args.limit is not None:
        # --limit은 전체 실행 전 소수 파일로 빠르게 확인할 때만 사용한다.
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        files = files[: args.limit]

    # 3) Mel과 mask를 분리해 저장할 출력 폴더를 준비한다.
    output_dir = args.output_dir or config.dataset.output_dir
    if not output_dir.is_absolute():
        output_dir = (config.project_root / output_dir).resolve()
    mel_dir = output_dir / "mels"
    mask_dir = output_dir / "masks"
    mel_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # manifest의 각 행, 실패 내역, 출력 shape 통계를 실행 중에 모은다.
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    shapes: Counter[str] = Counter()
    mask_shapes: Counter[str] = Counter()

    for index, source_path in enumerate(tqdm(files, desc="Creating log-Mels")):
        try:
            # 원본 길이와 sample rate는 manifest 기록용이다.
            source_info = sf.info(source_path)

            # 4) WAV를 16 kHz mono float32 waveform으로 읽는다.
            waveform = load_waveform(source_path, config.audio)

            # 5) 파일 전용 난수 생성기로 긴 음성의 random crop 위치를 결정한다.
            rng = random_generator_for_file(
                source_path,
                config.dataset.input_dir,
                config.audio.random_seed,
            )
            segment = make_training_segment(waveform, config.audio, rng)

            # 6) 고정 4초 구간을 log-Mel로 바꾸고 padding frame mask도 만든다.
            mel, frame_mask = segment_to_logmel(segment, config.audio, config.mel)

            # 7) Mel과 mask가 같은 파일명을 사용하도록 저장 경로를 만든다.
            filename = output_name(index, source_path, config.dataset.input_dir)
            destination = mel_dir / filename
            mask_destination = mask_dir / filename
            np.save(destination, mel, allow_pickle=False)
            np.save(mask_destination, frame_mask, allow_pickle=False)

            # 모든 결과가 예상한 shape인지 summary에서 한눈에 확인하기 위한 통계다.
            shapes[str(tuple(mel.shape))] += 1
            mask_shapes[str(tuple(frame_mask.shape))] += 1
            # crop 위치와 유효 frame 비율까지 기록해 나중에 각 결과를 추적할 수 있다.
            rows.append(
                {
                    "index": index,
                    "source_path": str(source_path),
                    "mel_path": str(destination),
                    "mask_path": str(mask_destination),
                    "source_duration_seconds": round(float(source_info.duration), 6),
                    "source_sample_rate": source_info.samplerate,
                    "source_start_seconds": round(
                        segment.source_start_sample / config.audio.sample_rate,
                        6,
                    ),
                    "valid_samples": segment.valid_samples,
                    "valid_frames": int(frame_mask.sum()),
                    "valid_frame_ratio": round(float(frame_mask.mean()), 6),
                    "shape": "x".join(map(str, mel.shape)),
                    "mask_shape": "x".join(map(str, frame_mask.shape)),
                    "dtype": str(mel.dtype),
                    "min": float(mel.min()),
                    "max": float(mel.max()),
                }
            )
        except Exception as exc:
            # 한 파일이 깨져 있어도 나머지는 계속 처리하고 실패 목록에 기록한다.
            failures.append({"path": str(source_path), "error": str(exc)})

    # 8) 원본 WAV, Mel, mask의 대응 관계를 CSV로 저장한다.
    manifest_path = output_dir / "manifest.csv"
    fieldnames = list(rows[0].keys()) if rows else ["index", "source_path", "mel_path"]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 9) 전체 성공/실패 개수와 실제 적용 설정을 JSON으로 남긴다.
    summary = {
        "input_dir": str(config.dataset.input_dir),
        "output_dir": str(output_dir),
        "requested_files": len(files),
        "processed_files": len(rows),
        "failed_files": failures,
        "shapes": dict(shapes),
        "mask_shapes": dict(mask_shapes),
        "crop_mode": config.audio.crop_mode,
        "random_seed": config.audio.random_seed,
        "inference_overlap_seconds": config.audio.inference_overlap_seconds,
        "normalization_range": [-1.0, 1.0],
        "config": {
            "sample_rate": config.audio.sample_rate,
            "duration_seconds": config.audio.duration_seconds,
            "target_samples": config.audio.target_samples,
            "n_mels": config.mel.n_mels,
            "n_fft": config.mel.n_fft,
            "hop_length": config.mel.hop_length,
            "top_db": config.mel.top_db,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nManifest: {manifest_path}")
    print(f"Summary:  {summary_path}")

    # 출력 파일을 먼저 남긴 뒤 실패 사실을 종료 코드로도 알린다.
    if failures:
        raise RuntimeError(f"Preprocessing failed for {len(failures)} file(s).")
    if args.limit is None and len(rows) < config.dataset.min_samples:
        raise RuntimeError(
            f"Only {len(rows):,} files were processed; minimum is {config.dataset.min_samples:,}."
        )


if __name__ == "__main__":
    main()
