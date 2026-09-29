"""
validation용 Real/Fake 음성 파일들의 점수를 계산하고,
ROC curve에서 EER(Equal Error Rate) 지점을 기준값으로 산출한다.
추가로 Real/Fake 점수 분포와 threshold를 이미지로 저장한다.

사용 예:
    python scripts/find_threshold.py \
        --real-dir data/val/real --fake-dir data/val/fake \
        --checkpoint checkpoints/best_model.pt --preprocess-config configs/preprocess.yaml \
        --eval-timesteps 20 60 100 150

sklearn과 matplotlib이 설치되어 있어야 한다 (pip install scikit-learn matplotlib).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 화면 창 없이 파일로만 저장 (서버/터미널 환경에서 안전)
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))  # pip install 없이 deepvoice_diffusion을 바로 찾기 위함
from scripts.score import load_model, score_audio_file  # noqa: E402
from deepvoice_diffusion.config import load_config as load_preprocess_config  # noqa: E402


def collect_scores(
    audio_dir: str, model, diffusion, device, audio_config, mel_config, eval_timesteps, aggregate
) -> list[float]:
    scores = []
    for audio_path in sorted(Path(audio_dir).glob("*.wav")):
        result = score_audio_file(
            str(audio_path), model, diffusion, device, audio_config, mel_config, eval_timesteps, aggregate
        )
        scores.append(result["file_score"])
    return scores


def find_eer_threshold(real_scores: list[float], fake_scores: list[float]) -> dict:
    """Real=0, Fake=1 라벨로 두고 ROC를 그려 EER 지점을 찾는다."""
    y_true = [0] * len(real_scores) + [1] * len(fake_scores)
    y_score = real_scores + fake_scores

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer_threshold = thresholds[eer_idx]
    eer_value = (fpr[eer_idx] + fnr[eer_idx]) / 2

    return {"threshold": float(eer_threshold), "eer": float(eer_value)}


def plot_score_distributions(
    real_scores: list[float],
    fake_scores: list[float],
    threshold: float,
    eer: float,
    output_path: Path,
) -> None:
    """Real/Fake 점수를 점 하나하나로 찍어서 threshold와 함께 이미지로 저장한다.

    표본이 적을 때(수십 개 이하)는 히스토그램보다 점을 직접 찍는 방식이
    "실제로 몇 개가 어디에 있는지"를 왜곡 없이 보여준다.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 점들이 서로 겹쳐 안 보이는 것을 막기 위해 y축에 살짝 무작위 흔들림(jitter)을 준다.
    rng = np.random.default_rng(0)
    real_y = 1.0 + rng.uniform(-0.08, 0.08, size=len(real_scores))
    fake_y = 0.0 + rng.uniform(-0.08, 0.08, size=len(fake_scores))

    fig, ax = plt.subplots(figsize=(8, 3.2))

    ax.scatter(
        real_scores, real_y, color="#1D9E75", s=45, alpha=0.85,
        label=f"Real (n={len(real_scores)})", edgecolors="white", linewidths=0.5,
    )
    ax.scatter(
        fake_scores, fake_y, color="#D85A30", s=45, alpha=0.85,
        label=f"Fake (n={len(fake_scores)})", edgecolors="white", linewidths=0.5,
    )

    # threshold를 점선으로 표시하고, 그 위에 threshold/EER 수치를 같이 써준다.
    ax.axvline(threshold, color="#555555", linestyle="--", linewidth=1.5)
    ax.text(
        threshold, 1.35, f"threshold = {threshold:.3f}\nEER = {eer * 100:.1f}%",
        ha="center", va="bottom", fontsize=9,
    )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fake", "Real"])
    ax.set_ylim(-0.4, 1.6)
    ax.set_xlabel("이상 점수 (anomaly score) — 낮음: Real 같음 / 높음: Fake 같음")
    ax.set_title("Real vs Fake 이상 점수 분포와 threshold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=str, required=True)
    parser.add_argument("--fake-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--preprocess-config", type=str, default="configs/preprocess.yaml")
    parser.add_argument("--aggregate", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument(
        "--eval-timesteps", type=int, nargs="+", default=[50, 150, 300, 450]
    )
    parser.add_argument(
        "--plot-output", type=str, default="artifacts/score_distribution.png",
        help="Real/Fake 점수 분포 그래프를 저장할 경로",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion, _ = load_model(args.checkpoint, device)
    preprocess_cfg = load_preprocess_config(args.preprocess_config)

    real_scores = collect_scores(
        args.real_dir, model, diffusion, device,
        preprocess_cfg.audio, preprocess_cfg.mel, args.eval_timesteps, args.aggregate,
    )
    fake_scores = collect_scores(
        args.fake_dir, model, diffusion, device,
        preprocess_cfg.audio, preprocess_cfg.mel, args.eval_timesteps, args.aggregate,
    )

    print(f"real 개수: {len(real_scores)}, fake 개수: {len(fake_scores)}")
    print(f"real 점수 평균/표준편차: {np.mean(real_scores):.5f} / {np.std(real_scores):.5f}")
    print(f"fake 점수 평균/표준편차: {np.mean(fake_scores):.5f} / {np.std(fake_scores):.5f}")

    result = find_eer_threshold(real_scores, fake_scores)
    print(f"\n권장 threshold: {result['threshold']:.5f}")
    print(f"해당 지점의 EER: {result['eer']*100:.2f}%")

    plot_path = Path(args.plot_output)
    plot_score_distributions(real_scores, fake_scores, result["threshold"], result["eer"], plot_path)
    print(f"\n분포 그래프 저장됨: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
