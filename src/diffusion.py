"""
Gaussian diffusion 유틸리티.

- 정방향(forward) 과정: 원본 x0에 단계 t만큼 노이즈를 섞어 x_t를 만든다.
- 학습 loss: 실제로 섞은 노이즈(eps)와 모델이 예측한 노이즈(eps_pred)의
  masked MSE.
- 점수 계산용 함수: 여러 timestep에서 노이즈 예측 오차를 구해 평균 낸
  "이상 점수"를 반환한다 (설계안 5.1절).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """values(스텝별 값들)에서 배치의 각 샘플에 해당하는 t 위치 값을 뽑아
    x_shape에 브로드캐스트 가능한 (B, 1, 1, 1) 형태로 반환한다."""
    batch_size = t.shape[0]
    out = values.gather(0, t.to(values.device))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


@dataclass
class GaussianDiffusion:
    timesteps: int = 500
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    device: str = "cpu"

    def __post_init__(self):
        betas = linear_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alpha_bars = alpha_bars.to(self.device)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars).to(self.device)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars).to(self.device)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """정방향 과정: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

        반환: (x_t, noise) — noise를 인자로 안 주면 새로 샘플링해서 같이 반환한다.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_1m_ab = extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)

        x_t = sqrt_ab * x0 + sqrt_1m_ab * noise
        return x_t, noise

    def training_losses(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """학습용 masked MSE loss 한 배치 계산.

        x0:   (B, 1, 80, T) 실제(Real) log-mel, 값 범위 [-1, 1]
        mask: (B, 1, 1, T) 또는 (B, 1, 80, T) — 1=실제 음성 프레임, 0=패딩.
              None이면 마스크 없이(전체 프레임) 계산.
        t:    (B,) 정수 timestep. None이면 배치마다 랜덤 샘플링.
        """
        batch_size = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=x0.device).long()

        x_t, noise = self.q_sample(x0, t)
        noise_pred = model(x_t, t)

        if mask is None:
            mask = torch.ones_like(x0)

        # 마스크를 곱해서 padding 프레임은 loss에 기여하지 않게 한다.
        sq_err = (noise_pred - noise) ** 2 * mask
        # 프레임 수로 정규화(마스크 합이 0인 극단적 경우 방지용 eps)
        loss = sq_err.sum() / (mask.sum() + 1e-8)
        return loss

    @torch.no_grad()
    def anomaly_score(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eval_timesteps: Optional[list[int]] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """설계안 5.1절: 여러 고정 timestep에서 노이즈 예측 오차를 구해 평균.

        x0:   (B, 1, 80, T) — 판정하려는 구간(4초) 배치
        반환: (B,) — 구간별 이상 점수 (낮을수록 Real에 가까움)
        """
        if eval_timesteps is None:
            eval_timesteps = [50, 150, 300, 450]
        if mask is None:
            mask = torch.ones_like(x0)

        if seed is not None:
            generator = torch.Generator(device=x0.device).manual_seed(seed)
        else:
            generator = None

        batch_size = x0.shape[0]
        scores = torch.zeros(batch_size, device=x0.device)

        for t_val in eval_timesteps:
            t = torch.full((batch_size,), t_val, device=x0.device, dtype=torch.long)
            noise = torch.randn(x0.shape, generator=generator, device=x0.device) if generator else torch.randn_like(x0)
            x_t, noise = self.q_sample(x0, t, noise=noise)
            noise_pred = model(x_t, t)

            sq_err = (noise_pred - noise) ** 2 * mask
            per_sample = sq_err.flatten(1).sum(1) / (mask.flatten(1).sum(1) + 1e-8)
            scores += per_sample

        return scores / len(eval_timesteps)
