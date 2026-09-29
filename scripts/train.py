"""
학습 스크립트 (진행바 + 빠른 테스트 옵션 추가 버전).

전제: scripts/preprocess.py를 실행하면 아래와 같은 결과물이 만들어진다.

    <output_dir>/
        mels/00000_xxxxxxxxxx.npy    # (1, 80, 251) float32, [-1, 1]
        masks/00000_xxxxxxxxxx.npy   # (1, 1, 251) float32, 1=실제 프레임
        manifest.csv                  # mel_path, mask_path, speaker_id, split 등 메타데이터

manifest의 split 컬럼에 따라 train/validation을 사용하고 test는 학습에서 제외한다.

빠른 확인용:
    python scripts/train.py --max-batches 5
위처럼 실행하면 각 epoch에서 5개 배치만 돌고 넘어가므로,
"코드가 끝까지 에러 없이 도는지"만 몇 초~1분 안에 확인할 수 있다.
학습 자체가 목적일 때는 --max-batches 없이 실행한다.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.diffusion import GaussianDiffusion  # noqa: E402
from src.model import UNet  # noqa: E402


def read_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class RealMelDataset(Dataset):
    """manifest.csv의 각 행이 가리키는 mel/mask .npy 쌍을 읽는다."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        mel = np.load(row["mel_path"])
        mask = np.load(row["mask_path"])
        return torch.from_numpy(mel), torch.from_numpy(mask)


def split_manifest_rows(
    rows: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """manifest에 기록된 화자 단위 분할을 그대로 사용한다."""
    required_columns = {"speaker_id", "split", "mel_path", "mask_path"}
    if not rows:
        raise ValueError("manifest.csv에 데이터가 없습니다.")

    missing_columns = required_columns - rows[0].keys()
    if missing_columns:
        raise ValueError(
            "manifest.csv에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing_columns))
        )

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]

    if not train_rows:
        raise ValueError("manifest.csv에 train 데이터가 없습니다.")
    if not val_rows:
        raise ValueError("manifest.csv에 validation 데이터가 없습니다.")
    if not test_rows:
        raise ValueError("manifest.csv에 test 데이터가 없습니다.")

    speaker_sets = {
        "train": {row["speaker_id"] for row in train_rows},
        "validation": {row["speaker_id"] for row in val_rows},
        "test": {row["speaker_id"] for row in test_rows},
    }
    if speaker_sets["train"] & speaker_sets["validation"]:
        raise ValueError("train과 validation에 같은 화자가 포함되어 있습니다.")
    if speaker_sets["train"] & speaker_sets["test"]:
        raise ValueError("train과 test에 같은 화자가 포함되어 있습니다.")
    if speaker_sets["validation"] & speaker_sets["test"]:
        raise ValueError("validation과 test에 같은 화자가 포함되어 있습니다.")

    return train_rows, val_rows, test_rows


def run_one_epoch(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    max_batches: int | None,
    train_mode: bool,
) -> float:
    """배치마다 tqdm 진행바를 갱신하면서 한 epoch(또는 --max-batches 만큼)을 돈다."""
    model.train() if train_mode else model.eval()

    label = "train" if train_mode else "val  "
    loss_sum, batch_count = 0.0, 0

    total = len(loader) if max_batches is None else min(max_batches, len(loader))
    progress = tqdm(
        loader,
        total=total,
        desc=f"[epoch {epoch + 1}/{total_epochs}] {label}",
        leave=False,
    )

    for mel, mask in progress:
        if max_batches is not None and batch_count >= max_batches:
            break

        mel, mask = mel.to(device), mask.to(device)

        if train_mode:
            loss = diffusion.training_losses(model, mel, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss = diffusion.training_losses(model, mel, mask)

        loss_sum += loss.item()
        batch_count += 1
        # 진행바 옆에 지금까지의 평균 loss를 실시간으로 표시한다.
        progress.set_postfix(avg_loss=f"{loss_sum / batch_count:.5f}")

    progress.close()
    return loss_sum / max(batch_count, 1)


def train(config_path: str, max_batches: int | None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if cfg["device"] == "cpu" or torch.cuda.is_available() else "cpu")

    manifest_path = Path(cfg["data_dir"]) / "manifest.csv"
    all_rows = read_manifest(manifest_path)
    train_rows, val_rows, test_rows = split_manifest_rows(all_rows)
    print(
        f"train samples: {len(train_rows)}, "
        f"val samples: {len(val_rows)}, test samples: {len(test_rows)}"
    )
    if max_batches is not None:
        print(f"[빠른 테스트 모드] epoch당 최대 {max_batches}개 배치만 실행합니다.")

    train_loader = DataLoader(
        RealMelDataset(train_rows), batch_size=cfg["batch_size"], shuffle=True, drop_last=True
    )
    val_loader = DataLoader(RealMelDataset(val_rows), batch_size=cfg["batch_size"], shuffle=False)

    model = UNet(
        base_channels=cfg["model"]["base_channels"],
        time_base_dim=cfg["model"]["time_base_dim"],
        time_dim=cfg["model"]["time_dim"],
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        device=str(device),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(cfg["epochs"]):
        train_loss = run_one_epoch(
            model, diffusion, train_loader, optimizer, device,
            epoch, cfg["epochs"], max_batches, train_mode=True,
        )
        scheduler.step()

        val_loss = run_one_epoch(
            model, diffusion, val_loader, optimizer, device,
            epoch, cfg["epochs"], max_batches, train_mode=False,
        )

        print(f"[epoch {epoch+1}/{cfg['epochs']}] train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state": model.state_dict(), "config": cfg, "epoch": epoch},
                ckpt_dir / "best_model.pt",
            )
            print(f"  -> best_model.pt 갱신 (val_loss={val_loss:.5f})")

    torch.save(
        {"model_state": model.state_dict(), "config": cfg, "epoch": cfg["epochs"] - 1},
        ckpt_dir / "last_model.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="epoch당 이 개수만큼만 배치를 돌고 넘어간다 (빠른 동작 확인용).",
    )
    args = parser.parse_args()
    train(args.config, args.max_batches)
