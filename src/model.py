"""
시간 조건부 2D U-Net — diffusion 노이즈 예측 네트워크 (epsilon predictor).

입력: log-mel spectrogram, shape (B, 1, 80, 251), 값 범위 [-1, 1]
     (forward 내부에서 시간축을 251 -> 256으로 zero-pad 후 처리하고,
      출력 시 다시 251로 잘라서 돌려준다)
출력: 입력과 같은 shape의 예측 노이즈 (B, 1, 80, 251)

설계안 문서의 3장 구조를 그대로 구현한다:
  Stem -> Down1 -> Down2 -> Bottleneck(+Attention) -> Up2 -> Up1 -> Head
  채널: 1 -> 64 -> 128 -> 256 -> 128 -> 64 -> 1
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 원본 시간축 길이(251)와 네트워크 내부에서 쓰는 패딩된 길이(256).
# 256 = 64 * 4 이므로 stride-2 다운샘플을 두 번 해도 나누어떨어진다.
ORIG_TIME_LEN = 251
PADDED_TIME_LEN = 256


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """DDPM/Transformer 스타일의 sinusoidal time embedding.

    timesteps: (B,) int64 또는 float 텐서
    반환: (B, dim)
    """
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:  # dim이 홀수면 0 하나 패딩
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    """sinusoidal embedding -> MLP(SiLU) 로 512차원 임베딩을 만든다."""

    def __init__(self, base_dim: int = 128, out_dim: int = 512):
        super().__init__()
        self.base_dim = base_dim
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_embedding(t, self.base_dim)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv -> (+time embedding, FiLM 방식) -> GroupNorm-SiLU-Conv -> residual."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, num_groups: int = 8):
        super().__init__()
        groups_in = min(num_groups, in_ch)
        groups_out = min(num_groups, out_ch)

        self.norm1 = nn.GroupNorm(groups_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # FiLM: time embedding으로부터 scale, shift 두 값을 만들어 feature map에 적용
        self.time_proj = nn.Linear(time_dim, out_ch * 2)

        self.norm2 = nn.GroupNorm(groups_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = h * (1 + scale) + shift

        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    """(B, C, H, W) 특징맵을 (H*W) 시퀀스로 펴서 표준 self-attention을 적용."""

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_ = self.norm(x).reshape(b, c, h * w).transpose(1, 2)  # (B, HW, C)
        out, _ = self.attn(h_, h_, h_, need_weights=False)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class UNet(nn.Module):
    """εθ(x_t, t) — 노이즈 예측 네트워크. 설계안 3.3절 표와 동일 구조."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        time_base_dim: int = 128,
        time_dim: int = 512,
    ):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4  # 64, 128, 256

        self.time_embedding = TimeEmbedding(time_base_dim, time_dim)

        # Stem
        self.stem = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        # Down 1: 80x256 -> 40x128
        self.down1_res1 = ResBlock(c1, c1, time_dim)
        self.down1_res2 = ResBlock(c1, c1, time_dim)
        self.down1_pool = Downsample(c1)
        self.down1_proj = nn.Conv2d(c1, c2, kernel_size=1)  # 채널 64->128

        # Down 2: 40x128 -> 20x64
        self.down2_res1 = ResBlock(c2, c2, time_dim)
        self.down2_res2 = ResBlock(c2, c2, time_dim)
        self.down2_pool = Downsample(c2)
        self.down2_proj = nn.Conv2d(c2, c3, kernel_size=1)  # 채널 128->256

        # Bottleneck: 20x64, 채널 256
        self.mid_res1 = ResBlock(c3, c3, time_dim)
        self.mid_attn = SelfAttention2D(c3)
        self.mid_res2 = ResBlock(c3, c3, time_dim)

        # Up 2: 20x64 -> 40x128, skip-concat with down1_res2 output(c2 채널) 이전에 채널 정합 필요
        self.up2_upsample = Upsample(c3)
        self.up2_proj = nn.Conv2d(c3, c2, kernel_size=1)  # 256->128, skip(128)과 concat 위해
        self.up2_res1 = ResBlock(c2 * 2, c2, time_dim)  # concat 후 256채널 -> 128
        self.up2_res2 = ResBlock(c2, c2, time_dim)

        # Up 1: 40x128 -> 80x256, skip-concat with down0(stem/down1_res 이전, c1 채널)
        self.up1_upsample = Upsample(c2)
        self.up1_proj = nn.Conv2d(c2, c1, kernel_size=1)  # 128->64
        self.up1_res1 = ResBlock(c1 * 2, c1, time_dim)
        self.up1_res2 = ResBlock(c1, c1, time_dim)

        # Head
        self.head_norm = nn.GroupNorm(min(8, c1), c1)
        self.head_conv = nn.Conv2d(c1, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 80, 251)
        t: (B,) 정수 timestep
        반환: (B, 1, 80, 251) — 예측 노이즈
        """
        _, _, freq_len, time_len = x.shape
        assert freq_len == 80, f"주파수(mel bin) 길이는 80이어야 합니다. got {freq_len}"

        pad_amount = PADDED_TIME_LEN - time_len
        if pad_amount < 0:
            raise ValueError(f"입력 시간 길이 {time_len}가 {PADDED_TIME_LEN}보다 큽니다.")
        # 시간축 오른쪽에 zero-pad (전처리 mask와는 별개의, 구조상 편의 패딩)
        x_padded = F.pad(x, (0, pad_amount))

        t_emb = self.time_embedding(t)

        # Stem
        h0 = self.stem(x_padded)  # (B, 64, 80, 256) — up1 skip용으로 보관

        # Down 1
        h1 = self.down1_res1(h0, t_emb)
        h1 = self.down1_res2(h1, t_emb)  # (B, 64, 80, 256) — up1 skip용
        h1_down = self.down1_proj(self.down1_pool(h1))  # (B, 128, 40, 128)

        # Down 2
        h2 = self.down2_res1(h1_down, t_emb)
        h2 = self.down2_res2(h2, t_emb)  # (B, 128, 40, 128) — up2 skip용
        h2_down = self.down2_proj(self.down2_pool(h2))  # (B, 256, 20, 64)

        # Bottleneck
        m = self.mid_res1(h2_down, t_emb)
        m = self.mid_attn(m)
        m = self.mid_res2(m, t_emb)  # (B, 256, 20, 64)

        # Up 2
        u2 = self.up2_proj(self.up2_upsample(m))  # (B, 128, 40, 128)
        u2 = torch.cat([u2, h2], dim=1)  # (B, 256, 40, 128)
        u2 = self.up2_res1(u2, t_emb)
        u2 = self.up2_res2(u2, t_emb)  # (B, 128, 40, 128)

        # Up 1
        u1 = self.up1_proj(self.up1_upsample(u2))  # (B, 64, 80, 256)
        u1 = torch.cat([u1, h1], dim=1)  # (B, 128, 80, 256)
        u1 = self.up1_res1(u1, t_emb)
        u1 = self.up1_res2(u1, t_emb)  # (B, 64, 80, 256)

        out = self.head_conv(F.silu(self.head_norm(u1)))  # (B, 1, 80, 256)

        # 패딩했던 시간축을 원래 길이로 잘라서 반환
        return out[:, :, :, :time_len]


if __name__ == "__main__":
    # 간단한 shape 확인용 (torch 설치된 환경에서 직접 실행)
    model = UNet()
    dummy_x = torch.randn(2, 1, 80, ORIG_TIME_LEN)
    dummy_t = torch.randint(0, 500, (2,))
    out = model(dummy_x, dummy_t)
    print("output shape:", out.shape)  # 기대값: (2, 1, 80, 251)
