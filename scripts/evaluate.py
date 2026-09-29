"""configs/evaluation.yaml 하나로 Real/Fake 평가를 실행한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from deepvoice_diffusion.config import load_config as load_preprocess_config  # noqa: E402
from scripts.find_threshold import (  # noqa: E402
    collect_scores,
    find_eer_threshold,
    plot_score_distributions,
    read_test_audio_paths,
)
from scripts.score import load_model  # noqa: E402


def project_path(value: str) -> Path:
    """상대 경로를 프로젝트 루트 기준 절대 경로로 바꾼다."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_evaluation_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    required = {
        "real_manifest",
        "fake_dir",
        "checkpoint",
        "preprocess_config",
        "eval_timesteps",
        "aggregate",
        "plot_output",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(
            "평가 설정에 필요한 항목이 없습니다: " + ", ".join(sorted(missing))
        )
    if config["aggregate"] not in {"mean", "max"}:
        raise ValueError("aggregate는 'mean' 또는 'max'여야 합니다.")
    if not config["eval_timesteps"]:
        raise ValueError("eval_timesteps는 하나 이상 지정해야 합니다.")

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="설정 파일을 사용해 모델을 평가합니다.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "evaluation.yaml",
    )
    args = parser.parse_args()

    config_path = project_path(str(args.config))
    config = load_evaluation_config(config_path)

    real_manifest = project_path(str(config["real_manifest"]))
    fake_dir = project_path(str(config["fake_dir"]))
    checkpoint = project_path(str(config["checkpoint"]))
    preprocess_config = project_path(str(config["preprocess_config"]))
    plot_output = project_path(str(config["plot_output"]))

    if not fake_dir.is_dir():
        raise FileNotFoundError(
            f"Fake 음성 폴더가 없습니다: {fake_dir}\n"
            "configs/evaluation.yaml의 fake_dir을 실제 폴더로 바꿔주세요."
        )

    fake_paths = sorted(
        path
        for path in fake_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    if not fake_paths:
        raise FileNotFoundError(f"Fake WAV 파일을 찾지 못했습니다: {fake_dir}")

    real_paths = read_test_audio_paths(real_manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion, _ = load_model(str(checkpoint), device)
    preprocess = load_preprocess_config(preprocess_config)

    eval_timesteps = [int(value) for value in config["eval_timesteps"]]
    aggregate = str(config["aggregate"])

    real_scores = collect_scores(
        real_paths,
        model,
        diffusion,
        device,
        preprocess.audio,
        preprocess.mel,
        eval_timesteps,
        aggregate,
        desc="Real 평가",
    )
    fake_scores = collect_scores(
        fake_paths,
        model,
        diffusion,
        device,
        preprocess.audio,
        preprocess.mel,
        eval_timesteps,
        aggregate,
        desc="Fake 평가",
    )

    print(f"real 개수: {len(real_scores)}, fake 개수: {len(fake_scores)}")
    print(
        f"real 점수 평균/표준편차: "
        f"{np.mean(real_scores):.5f} / {np.std(real_scores):.5f}"
    )
    print(
        f"fake 점수 평균/표준편차: "
        f"{np.mean(fake_scores):.5f} / {np.std(fake_scores):.5f}"
    )

    result = find_eer_threshold(real_scores, fake_scores)
    print(f"\n권장 threshold: {result['threshold']:.5f}")
    print(f"해당 지점의 EER: {result['eer'] * 100:.2f}%")

    plot_score_distributions(
        real_scores,
        fake_scores,
        result["threshold"],
        result["eer"],
        plot_output,
    )
    print(f"\n분포 그래프 저장됨: {plot_output}")


if __name__ == "__main__":
    main()
