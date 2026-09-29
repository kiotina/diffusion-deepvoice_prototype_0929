# 음성 전처리 담당

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .config import AudioConfig, MelConfig


@dataclass(frozen=True)
class AudioSegment:
    """모델 입력 한 구간과 그 구간이 원본의 어디에서 왔는지 묶어 보관한다."""

    # 항상 target_samples 길이인 실제 모델 입력
    waveform: np.ndarray
    # 원본 음성은 1, zero padding은 0으로 표시한 sample 단위 mask
    sample_mask: np.ndarray
    # 긴 원본에서 이 구간을 잘라내기 시작한 sample 위치
    source_start_sample: int
    # crop 또는 padding하기 전 원본 waveform의 전체 sample 수
    source_samples: int

    @property
    def valid_samples(self) -> int:
        return int(self.sample_mask.sum())


def load_waveform(path: str | Path, config: AudioConfig) -> np.ndarray:
    """WAV를 모델 전처리의 공통 형식인 16 kHz mono float32로 읽는다."""
    # sr을 지정하면 입력 sample rate가 다를 때 librosa가 16 kHz로 변환한다.
    # mono=True는 스테레오 입력을 한 채널로 합친다.
    waveform, _ = librosa.load(
        path,
        sr=config.sample_rate,
        mono=True,
        dtype=np.float32,
    )
    # 빈 파일과 NaN/무한대가 포함된 파일은 이후 계산을 망가뜨리므로 즉시 중단한다.
    if waveform.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"Non-finite samples found: {path}")
    # 현재 설정은 false다. 녹음 환경과 원래의 무음 구간을 보존하기 위해
    # 명시적으로 켜기 전에는 앞뒤 무음을 제거하지 않는다.
    if config.trim_silence:
        waveform, _ = librosa.effects.trim(waveform, top_db=config.trim_top_db)
    return waveform.astype(np.float32, copy=False)


def make_segment(
    waveform: np.ndarray,
    target_samples: int,
    *,
    start_sample: int | None = None,
) -> AudioSegment:
    """고정 길이 구간과 실제 음성/zero padding을 구분하는 mask를 만든다."""
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform, got shape {waveform.shape}")
    if waveform.size == 0:
        raise ValueError("Cannot segment an empty waveform")
    if target_samples < 1:
        raise ValueError("target_samples must be positive")

    source_samples = waveform.shape[0]
    if source_samples >= target_samples:
        # 원본이 충분히 길면 padding 없이 지정된 시작점에서 정확히 4초를 자른다.
        maximum_start = source_samples - target_samples
        if start_sample is None:
            # 시작점이 없으면 검사 이미지처럼 재현성이 필요한 용도로 중앙을 사용한다.
            start_sample = maximum_start // 2
        if not 0 <= start_sample <= maximum_start:
            raise ValueError(
                f"start_sample must be between 0 and {maximum_start}, got {start_sample}"
            )
        segment = waveform[start_sample : start_sample + target_samples]
        # 잘라낸 4초 전체가 실제 음성이므로 모든 sample이 유효하다.
        mask = np.ones(target_samples, dtype=np.float32)
    else:
        # 원본이 4초보다 짧으면 부족한 길이를 좌우에 가능한 한 균등하게 나눈다.
        start_sample = 0
        difference = target_samples - source_samples
        left = difference // 2
        right = difference - left
        segment = np.pad(waveform, (left, right))
        # waveform과 똑같은 위치에 0을 채워 padding 영역을 표시한다.
        mask = np.pad(
            np.ones(source_samples, dtype=np.float32),
            (left, right),
        )

    return AudioSegment(
        waveform=segment.astype(np.float32, copy=False),
        sample_mask=mask,
        source_start_sample=start_sample,
        source_samples=source_samples,
    )


def make_training_segment(
    waveform: np.ndarray,
    config: AudioConfig,
    rng: np.random.Generator,
) -> AudioSegment:
    """학습용 구간을 만든다. 긴 음성은 설정에 따라 random 또는 center crop한다."""
    start_sample: int | None = None
    if waveform.shape[0] > config.target_samples:
        maximum_start = waveform.shape[0] - config.target_samples
        if config.crop_mode == "random":
            # 끝 위치도 선택 후보에 포함하기 위해 상한에 1을 더한다.
            start_sample = int(rng.integers(0, maximum_start + 1))
        else:
            start_sample = maximum_start // 2
    return make_segment(
        waveform,
        config.target_samples,
        start_sample=start_sample,
    )


def make_inference_segments(
    waveform: np.ndarray,
    config: AudioConfig,
) -> list[AudioSegment]:
    """긴 사용자 음성 전체를 겹치는 4초 모델 입력들로 분할한다."""
    target = config.target_samples
    if waveform.shape[0] <= target:
        # 4초 이하라면 한 구간만 만들고, 짧은 부분은 mask와 함께 padding한다.
        return [make_segment(waveform, target)]

    last_start = waveform.shape[0] - target
    # 현재 hop은 2초이므로 0, 2, 4, ...초 위치에서 4초 창을 만든다.
    starts = list(range(0, last_start + 1, config.inference_hop_samples))
    if starts[-1] != last_start:
        # 마지막 창이 음성 끝에 정확히 닿지 않으면 끝에 맞춘 창을 하나 더 추가한다.
        starts.append(last_start)
    return [
        make_segment(waveform, target, start_sample=start)
        for start in starts
    ]


def fit_duration(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    """기존 호출 호환용 함수: 중앙 crop 또는 대칭 padding한 waveform만 반환한다."""
    return make_segment(waveform, target_samples).waveform


def waveform_to_logmel(
    waveform: np.ndarray,
    audio_config: AudioConfig,
    mel_config: MelConfig,
) -> np.ndarray:
    """4초 waveform을 (1, 80, 251) 크기의 정규화된 log-Mel로 바꾼다."""
    # power=2.0이므로 진폭이 아니라 에너지를 표현하는 power spectrogram이다.
    # center=True는 각 frame의 시점을 분석 창 중앙으로 맞춘다.
    mel_power = librosa.feature.melspectrogram(
        y=waveform,
        sr=audio_config.sample_rate,
        n_fft=mel_config.n_fft,
        hop_length=mel_config.hop_length,
        win_length=mel_config.win_length,
        n_mels=mel_config.n_mels,
        fmin=mel_config.f_min,
        fmax=mel_config.f_max,
        power=2.0,
        center=True,
    )
    # 각 파일 안에서 가장 강한 에너지를 0 dB로 두고,
    # 그보다 80 dB 이상 작은 값은 -80 dB로 제한한다.
    # 이 파일별 기준은 녹음 환경 보존 논의에서 추후 다시 검토할 대상이다.
    logmel_db = librosa.power_to_db(
        mel_power,
        ref=np.max,
        top_db=mel_config.top_db,
    )
    # diffusion 모델이 다루기 쉬운 범위가 되도록 -80~0 dB를 -1~1로 선형 변환한다.
    normalized = 2.0 * (logmel_db + mel_config.top_db) / mel_config.top_db - 1.0
    normalized = np.clip(normalized, -1.0, 1.0).astype(np.float32)
    return normalized[np.newaxis, ...]


def sample_mask_to_frame_mask(
    sample_mask: np.ndarray,
    frame_count: int,
    mel_config: MelConfig,
) -> np.ndarray:
    """sample 단위 mask를 (1, 1, 251) Mel 시간-frame mask로 바꾼다."""
    # center=True인 Mel의 t번째 frame 중심은 대략 t * hop_length sample에 있다.
    centers = np.arange(frame_count, dtype=np.int64) * mel_config.hop_length
    # 마지막 frame 중심이 배열 끝과 같아질 수 있으므로 유효 index로 제한한다.
    centers = np.minimum(centers, sample_mask.shape[0] - 1)
    frame_mask = sample_mask[centers].astype(np.float32, copy=False)
    # (1, 1, time)으로 만들면 학습 시 (batch, channel, mel, time)에 broadcast할 수 있다.
    return frame_mask[np.newaxis, np.newaxis, :]


def segment_to_logmel(
    segment: AudioSegment,
    audio_config: AudioConfig,
    mel_config: MelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """하나의 고정 길이 구간에서 Mel과 그에 대응하는 frame mask를 함께 만든다."""
    mel = waveform_to_logmel(segment.waveform, audio_config, mel_config)
    frame_mask = sample_mask_to_frame_mask(
        segment.sample_mask,
        mel.shape[-1],
        mel_config,
    )
    return mel, frame_mask


def audio_file_to_logmel(
    path: str | Path,
    audio_config: AudioConfig,
    mel_config: MelConfig,
) -> np.ndarray:
    """데이터 검사와 시각화용으로 항상 중앙 구간의 Mel을 만든다."""
    waveform = load_waveform(path, audio_config)
    segment = make_segment(waveform, audio_config.target_samples)
    mel, _ = segment_to_logmel(segment, audio_config, mel_config)
    return mel
