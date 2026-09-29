"""
추론(점수 계산) 스크립트 — 실제 deepvoice_diffusion 패키지 함수에 맞춰 수정된 버전.

사용 예:
    python scripts/score.py --audio some_voice.wav \
        --checkpoint checkpoints/best_model.pt \
        --preprocess-config configs/preprocess.yaml \
        --threshold 0.0123 --aggregate mean

deepvoice_diffusion 패키지가 설치되어 있어야 한다 (팀원 저장소 루트에서
`pip install -e ".[dev]"` 실행 후 사용).
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))  # pip install 없이 deepvoice_diffusion을 바로 찾기 위함
from src.diffusion import GaussianDiffusion  # noqa: E402
from src.model import UNet  # noqa: E402

# 전처리 패키지 (audio.py) 에서 그대로 가져온다.
from deepvoice_diffusion.audio import (  # noqa: E402
    load_waveform,
    make_inference_segments,
    segment_to_logmel,
)
from deepvoice_diffusion.config import load_config as load_preprocess_config  # noqa: E402


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    model = UNet(
        base_channels=cfg["model"]["base_channels"],
        time_base_dim=cfg["model"]["time_base_dim"],
        time_dim=cfg["model"]["time_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        device=str(device),
    )
    return model, diffusion, cfg


def score_audio_file(
    audio_path: str,
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    device: torch.device,
    audio_config,  # deepvoice_diffusion.config.AudioConfig
    mel_config,    # deepvoice_diffusion.config.MelConfig
    eval_timesteps: list[int],
    aggregate: str = "mean",
    seed: int = 0,
) -> dict:
    """음성 파일 하나를 팀원의 make_inference_segments()로 4초 구간들로 나눈 뒤
    각 구간을 log-mel로 바꿔 이상 점수를 매기고, 파일 전체 점수를 반환한다."""

    waveform = load_waveform(audio_path, audio_config)
    segments = make_inference_segments(waveform, audio_config)  # AudioSegment 리스트

    segment_scores = []
    for segment in segments:
        mel, frame_mask = segment_to_logmel(segment, audio_config, mel_config)
        # mel: (1, 80, 251), frame_mask: (1, 1, 251) -> 배치 차원 추가
        mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)         # (1, 1, 80, 251)
        mask_t = torch.from_numpy(frame_mask).unsqueeze(0).to(device)  # (1, 1, 1, 251)

        score = diffusion.anomaly_score(
            model, mel_t, mask_t, eval_timesteps=eval_timesteps, seed=seed
        )
        segment_scores.append(score.item())

    if aggregate == "mean":
        file_score = sum(segment_scores) / len(segment_scores)
    elif aggregate == "max":
        file_score = max(segment_scores)
    else:
        raise ValueError(f"알 수 없는 aggregate 방식: {aggregate}")

    return {"segment_scores": segment_scores, "file_score": file_score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--preprocess-config",
        type=str,
        default="configs/preprocess.yaml",
        help="audio/mel 설정을 담고 있는 팀원의 전처리 설정 파일",
    )
    parser.add_argument("--threshold", type=float, default=None, help="지정하면 Real/Fake 판정까지 출력")
    parser.add_argument("--aggregate", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument(
        "--eval-timesteps", type=int, nargs="+", default=[50, 150, 300, 450]
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion, _ = load_model(args.checkpoint, device)

    # audio_config / mel_config는 전처리와 완전히 같은 설정을 써야 한다
    # (다른 sample_rate, n_mels 등으로 만들면 모델 입력과 안 맞음).
    preprocess_cfg = load_preprocess_config(args.preprocess_config)

    result = score_audio_file(
        args.audio,
        model,
        diffusion,
        device,
        preprocess_cfg.audio,
        preprocess_cfg.mel,
        args.eval_timesteps,
        args.aggregate,
    )

    print(f"구간별 점수: {['%.5f' % s for s in result['segment_scores']]}")
    print(f"파일 전체 점수({args.aggregate}): {result['file_score']:.5f}")

    if args.threshold is not None:
        verdict = "Fake" if result["file_score"] >= args.threshold else "Real"
        print(f"판정 (threshold={args.threshold}): {verdict}")


if __name__ == "__main__":
    main()
