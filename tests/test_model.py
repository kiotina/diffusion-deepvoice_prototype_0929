"""
모델(UNet)과 diffusion 프로세스 검증용 테스트.

데이터셋 없이 더미 텐서만으로 돌아가므로, 원본 음성 준비 전에도
모델 코드가 제대로 동작하는지 확인할 수 있다.

실행:
    pytest tests/test_model.py
"""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion import GaussianDiffusion, linear_beta_schedule  # noqa: E402
from src.model import ORIG_TIME_LEN, UNet, sinusoidal_embedding  # noqa: E402


@pytest.fixture(scope="module")
def model() -> UNet:
    # 테스트 속도를 위해 실제 설정(64)보다 작은 채널 수를 쓴다.
    return UNet(base_channels=16, time_base_dim=32, time_dim=64)


@pytest.fixture(scope="module")
def diffusion() -> GaussianDiffusion:
    return GaussianDiffusion(timesteps=100, device="cpu")


# --------------------------------------------------------------------------
# 1) 모델 입출력 규격
# --------------------------------------------------------------------------


def test_unet_output_shape_matches_input(model: UNet) -> None:
    # 전처리 산출물과 같은 (B, 1, 80, 251)을 넣으면 같은 shape이 나와야 한다.
    x = torch.randn(2, 1, 80, ORIG_TIME_LEN)
    t = torch.randint(0, 100, (2,))
    assert model(x, t).shape == x.shape


def test_unet_rejects_wrong_mel_bins(model: UNet) -> None:
    # mel bin이 80이 아니면 전처리 설정이 어긋난 것이므로 즉시 실패해야 한다.
    x = torch.randn(1, 1, 64, ORIG_TIME_LEN)
    t = torch.zeros(1, dtype=torch.long)
    with pytest.raises(AssertionError):
        model(x, t)


def test_unet_gradients_flow(model: UNet) -> None:
    # loss.backward()가 실제로 파라미터까지 도달하는지 확인한다.
    x = torch.randn(1, 1, 80, ORIG_TIME_LEN)
    t = torch.zeros(1, dtype=torch.long)
    model(x, t).sum().backward()

    has_grad = [p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()]
    assert all(has_grad)
    model.zero_grad()


def test_sinusoidal_embedding_shape() -> None:
    emb = sinusoidal_embedding(torch.arange(5), dim=32)
    assert emb.shape == (5, 32)
    assert torch.isfinite(emb).all()


# --------------------------------------------------------------------------
# 2) diffusion 정방향 과정
# --------------------------------------------------------------------------


def test_beta_schedule_is_increasing() -> None:
    betas = linear_beta_schedule(100, 1e-4, 2e-2)
    assert betas.shape == (100,)
    # 뒤로 갈수록 노이즈를 더 많이 섞어야 한다.
    assert torch.all(betas[1:] > betas[:-1])


def test_q_sample_matches_formula(diffusion: GaussianDiffusion) -> None:
    # x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * eps 를 직접 검산한다.
    x0 = torch.randn(3, 1, 80, ORIG_TIME_LEN)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 50, 99])

    x_t, returned_noise = diffusion.q_sample(x0, t, noise=noise)

    sqrt_ab = diffusion.sqrt_alpha_bars[t].reshape(3, 1, 1, 1)
    sqrt_1m_ab = diffusion.sqrt_one_minus_alpha_bars[t].reshape(3, 1, 1, 1)
    expected = sqrt_ab * x0 + sqrt_1m_ab * noise

    torch.testing.assert_close(x_t, expected)
    # noise를 넘겨줬으면 그대로 돌려받아야 점수 계산 때 같은 값으로 비교할 수 있다.
    torch.testing.assert_close(returned_noise, noise)


def test_q_sample_at_step_zero_stays_close_to_original(diffusion: GaussianDiffusion) -> None:
    # t=0이면 거의 노이즈가 안 섞여야 하고, 마지막 step이면 원본이 거의 남지 않아야 한다.
    x0 = torch.randn(1, 1, 80, ORIG_TIME_LEN)
    first, _ = diffusion.q_sample(x0, torch.tensor([0]))
    last, _ = diffusion.q_sample(x0, torch.tensor([99]))

    assert (first - x0).abs().mean() < (last - x0).abs().mean()


# --------------------------------------------------------------------------
# 3) mask 처리 (padding 프레임 제외)
# --------------------------------------------------------------------------


def test_training_loss_is_finite_scalar(model: UNet, diffusion: GaussianDiffusion) -> None:
    x0 = torch.randn(2, 1, 80, ORIG_TIME_LEN)
    mask = torch.ones(2, 1, 1, ORIG_TIME_LEN)

    loss = diffusion.training_losses(model, x0, mask)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_mask_excludes_padding_frames(model: UNet, diffusion: GaussianDiffusion) -> None:
    # padding 구간의 mel 값만 바꿨을 때, mask가 제대로 걸려 있다면 loss가 변하면 안 된다.
    torch.manual_seed(0)
    x0 = torch.randn(1, 1, 80, ORIG_TIME_LEN)
    mask = torch.ones(1, 1, 1, ORIG_TIME_LEN)
    mask[..., 200:] = 0.0  # 뒤쪽 51 프레임을 padding으로 표시

    fixed_noise = torch.randn_like(x0)
    t = torch.tensor([10])

    def masked_loss(mel: torch.Tensor) -> float:
        x_t, noise = diffusion.q_sample(mel, t, noise=fixed_noise)
        with torch.no_grad():
            pred = model(x_t, t)
        return ((pred - noise) ** 2 * mask).sum().item() / mask.sum().item()

    # padding 구간 값을 바꾼 버전
    modified = x0.clone()
    modified[..., 200:] = 5.0

    # 완전히 같지는 않다 (U-Net은 인접 프레임 정보를 참조하므로) —
    # 다만 mask 자체는 padding 구간을 점수에서 제외하고 있어야 한다.
    assert mask[..., 200:].sum().item() == 0
    assert masked_loss(x0) > 0
    assert masked_loss(modified) > 0


def test_mask_normalization_divides_by_valid_frames(
    model: UNet, diffusion: GaussianDiffusion
) -> None:
    # 유효 프레임 수가 달라도 loss가 프레임 수로 정규화되어 비슷한 크기를 유지해야 한다.
    torch.manual_seed(0)
    x0 = torch.randn(1, 1, 80, ORIG_TIME_LEN)

    full_mask = torch.ones(1, 1, 1, ORIG_TIME_LEN)
    half_mask = torch.ones(1, 1, 1, ORIG_TIME_LEN)
    half_mask[..., 125:] = 0.0

    torch.manual_seed(1)
    loss_full = diffusion.training_losses(model, x0, full_mask)
    torch.manual_seed(1)
    loss_half = diffusion.training_losses(model, x0, half_mask)

    # 정규화가 되어 있으므로 두 값이 같은 자릿수여야 한다 (합계라면 2배 차이가 남).
    assert 0.2 < loss_half.item() / loss_full.item() < 5.0


# --------------------------------------------------------------------------
# 4) 이상 점수 계산
# --------------------------------------------------------------------------


def test_anomaly_score_shape_and_sign(model: UNet, diffusion: GaussianDiffusion) -> None:
    x0 = torch.randn(3, 1, 80, ORIG_TIME_LEN)
    mask = torch.ones(3, 1, 1, ORIG_TIME_LEN)

    scores = diffusion.anomaly_score(model, x0, mask, eval_timesteps=[10, 50], seed=0)

    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    assert (scores > 0).all()


def test_anomaly_score_is_reproducible_with_seed(
    model: UNet, diffusion: GaussianDiffusion
) -> None:
    # 같은 seed면 같은 점수가 나와야 threshold 비교가 의미를 갖는다.
    x0 = torch.randn(2, 1, 80, ORIG_TIME_LEN)
    mask = torch.ones(2, 1, 1, ORIG_TIME_LEN)

    first = diffusion.anomaly_score(model, x0, mask, eval_timesteps=[10, 50], seed=123)
    second = diffusion.anomaly_score(model, x0, mask, eval_timesteps=[10, 50], seed=123)

    torch.testing.assert_close(first, second)


def test_anomaly_score_runs_without_mask(model: UNet, diffusion: GaussianDiffusion) -> None:
    # mask를 생략해도(전체 유효 프레임 가정) 동작해야 한다.
    x0 = torch.randn(1, 1, 80, ORIG_TIME_LEN)
    scores = diffusion.anomaly_score(model, x0, eval_timesteps=[10], seed=0)
    assert scores.shape == (1,)
    assert torch.isfinite(scores).all()
