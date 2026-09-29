# 전처리 설정 관리

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    """원본과 출력 경로, 사용할 데이터 개수 조건."""
    input_dir: Path
    output_dir: Path
    min_samples: int
    max_samples: int | None


@dataclass(frozen=True)
class AudioConfig:
    """waveform 로딩, 길이 통일, 구간 분할에 필요한 설정."""
    sample_rate: int
    duration_seconds: float
    crop_mode: str
    random_seed: int
    inference_overlap_seconds: float
    trim_silence: bool
    trim_top_db: float

    @property
    def target_samples(self) -> int:
        # 16,000 Hz × 4초 = 모델 입력 한 구간의 64,000 sample
        return round(self.sample_rate * self.duration_seconds)

    @property
    def inference_hop_samples(self) -> int:
        # 4초 창에 2초 overlap을 주면 다음 창은 2초(32,000 sample) 뒤에서 시작한다.
        return round(
            self.sample_rate
            * (self.duration_seconds - self.inference_overlap_seconds)
        )


@dataclass(frozen=True)
class MelConfig:
    """STFT와 Mel-spectrogram 변환에 필요한 설정."""
    n_fft: int
    hop_length: int
    win_length: int
    n_mels: int
    f_min: float
    f_max: float
    top_db: float


@dataclass(frozen=True)
class PrototypeConfig:
    """전처리에 필요한 모든 설정과 프로젝트 기준 경로."""
    dataset: DatasetConfig
    audio: AudioConfig
    mel: MelConfig
    project_root: Path


def _project_path(value: str, project_root: Path) -> Path:
    """상대 경로를 프로젝트 루트 기준 절대 경로로 변환한다."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def _require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """YAML의 필수 section이 존재하고 key-value 구조인지 확인한다."""
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid '{name}' section in config")
    return section


def load_config(config_path: str | Path) -> PrototypeConfig:
    """YAML을 읽어 타입이 명확한 설정 객체로 변환하고 검증한다."""
    config_path = Path(config_path).resolve()
    # configs/preprocess.yaml의 두 단계 위가 프로젝트 루트다.
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    dataset_raw = _require_section(raw, "dataset")
    audio_raw = _require_section(raw, "audio")
    mel_raw = _require_section(raw, "mel")
    max_samples = dataset_raw.get("max_samples")

    # YAML 문자열과 숫자를 명시적으로 원하는 Python 타입으로 변환한다.
    config = PrototypeConfig(
        dataset=DatasetConfig(
            input_dir=_project_path(str(dataset_raw["input_dir"]), project_root),
            output_dir=_project_path(str(dataset_raw["output_dir"]), project_root),
            min_samples=int(dataset_raw["min_samples"]),
            max_samples=None if max_samples is None else int(max_samples),
        ),
        audio=AudioConfig(
            sample_rate=int(audio_raw["sample_rate"]),
            duration_seconds=float(audio_raw["duration_seconds"]),
            crop_mode=str(audio_raw["crop_mode"]),
            random_seed=int(audio_raw["random_seed"]),
            inference_overlap_seconds=float(audio_raw["inference_overlap_seconds"]),
            trim_silence=bool(audio_raw["trim_silence"]),
            trim_top_db=float(audio_raw["trim_top_db"]),
        ),
        mel=MelConfig(
            n_fft=int(mel_raw["n_fft"]),
            hop_length=int(mel_raw["hop_length"]),
            win_length=int(mel_raw["win_length"]),
            n_mels=int(mel_raw["n_mels"]),
            f_min=float(mel_raw["f_min"]),
            f_max=float(mel_raw["f_max"]),
            top_db=float(mel_raw["top_db"]),
        ),
        project_root=project_root,
    )
    # 파일 처리를 시작하기 전에 잘못된 조합을 차단한다.
    _validate(config)
    return config


def _validate(config: PrototypeConfig) -> None:
    """실행은 가능하지만 의미가 없거나 오류를 낼 설정 조합을 차단한다."""
    if config.dataset.min_samples < 1:
        raise ValueError("dataset.min_samples must be positive")
    if config.dataset.max_samples is not None:
        if config.dataset.max_samples < config.dataset.min_samples:
            raise ValueError("dataset.max_samples cannot be smaller than min_samples")
    if config.audio.sample_rate < 1 or config.audio.duration_seconds <= 0:
        raise ValueError("sample_rate and duration_seconds must be positive")
    if config.audio.crop_mode not in {"random", "center"}:
        raise ValueError("audio.crop_mode must be 'random' or 'center'")
    if not 0 <= config.audio.inference_overlap_seconds < config.audio.duration_seconds:
        raise ValueError(
            "inference_overlap_seconds must be at least 0 and shorter than duration_seconds"
        )
    if config.mel.f_max > config.audio.sample_rate / 2:
        # sample rate의 절반보다 높은 주파수는 디지털 음성에 표현될 수 없다.
        raise ValueError("mel.f_max cannot exceed the Nyquist frequency")
    if config.mel.win_length > config.mel.n_fft:
        raise ValueError("mel.win_length cannot exceed mel.n_fft")
