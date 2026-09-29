"""Real 음성을 화자 기준으로 train/validation/test에 분리한다.

이 모듈은 원본 파일을 복사하거나 이동하지 않는다. 각 분할에 속하는
WAV 경로, JSON 경로, 화자 ID를 파이썬 자료구조로 반환한다.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


SpeakerSample = dict[str, Path | str]
SpeakerGroups = dict[str, list[SpeakerSample]]
SpeakerSplit = dict[str, list[SpeakerSample]]


# 현재 프로젝트에서 사용하는 두 JSON 형식의 화자 ID 위치다.
DEFAULT_SPEAKER_ID_PATHS: tuple[tuple[str, ...], ...] = (
    ("녹음자정보", "recorderId"),
    ("기본정보", "NumberOfSpeaker"),
)


def _files_by_stem(root: Path, suffix: str) -> dict[str, Path]:
    """하위 폴더의 파일을 확장자를 제외한 파일명으로 색인한다."""
    files: dict[str, Path] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != suffix:
            continue

        if path.stem in files:
            raise ValueError(
                f"같은 이름의 {suffix} 파일이 두 개 이상 있습니다.\n"
                f"첫 번째 파일: {files[path.stem]}\n"
                f"두 번째 파일: {path}"
            )

        files[path.stem] = path.resolve()

    return files


def match_audio_and_metadata(real_dir: str | Path) -> list[tuple[Path, Path]]:
    """Real 폴더에서 이름이 같은 WAV와 JSON을 일대일로 연결한다."""
    real_dir = Path(real_dir).expanduser().resolve()
    if not real_dir.is_dir():
        raise FileNotFoundError(f"Real 데이터 폴더가 없습니다: {real_dir}")

    wav_by_stem = _files_by_stem(real_dir, ".wav")
    json_by_stem = _files_by_stem(real_dir, ".json")

    if not wav_by_stem:
        raise FileNotFoundError(f"WAV 파일을 찾지 못했습니다: {real_dir}")
    if not json_by_stem:
        raise FileNotFoundError(f"JSON 파일을 찾지 못했습니다: {real_dir}")

    wav_without_json = sorted(wav_by_stem.keys() - json_by_stem.keys())
    json_without_wav = sorted(json_by_stem.keys() - wav_by_stem.keys())

    if wav_without_json or json_without_wav:
        messages: list[str] = []
        if wav_without_json:
            preview = ", ".join(wav_without_json[:5])
            messages.append(f"JSON이 없는 WAV {len(wav_without_json)}개: {preview}")
        if json_without_wav:
            preview = ", ".join(json_without_wav[:5])
            messages.append(f"WAV가 없는 JSON {len(json_without_wav)}개: {preview}")
        raise ValueError(
            "WAV와 JSON이 일대일로 대응하지 않습니다.\n" + "\n".join(messages)
        )

    return [
        (wav_by_stem[stem], json_by_stem[stem])
        for stem in sorted(wav_by_stem)
    ]


def _nested_value(data: dict[str, Any], key_path: tuple[str, ...]) -> Any:
    """중첩된 JSON 객체에서 지정한 key 경로의 값을 읽는다."""
    value: Any = data
    for key in key_path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def read_speaker_id(
    json_file: str | Path,
    speaker_id_paths: tuple[tuple[str, ...], ...] = DEFAULT_SPEAKER_ID_PATHS,
) -> str:
    """지원하는 JSON key 경로 중 하나에서 화자 ID를 읽는다."""
    json_file = Path(json_file)
    with json_file.open("r", encoding="utf-8") as handle:
        json_data = json.load(handle)

    if not isinstance(json_data, dict):
        raise ValueError(f"JSON 최상위 값이 객체가 아닙니다: {json_file}")

    for key_path in speaker_id_paths:
        speaker_id = _nested_value(json_data, key_path)
        if speaker_id is not None and str(speaker_id).strip():
            return str(speaker_id).strip()

    checked_paths = ", ".join(".".join(path) for path in speaker_id_paths)
    raise ValueError(
        f"화자 ID를 찾지 못했습니다: {json_file}\n"
        f"확인한 JSON 경로: {checked_paths}"
    )


def group_audio_by_speaker(
    real_dir: str | Path,
    speaker_id_paths: tuple[tuple[str, ...], ...] = DEFAULT_SPEAKER_ID_PATHS,
) -> SpeakerGroups:
    """Real 폴더의 WAV를 JSON에 기록된 화자 ID별로 묶는다."""
    grouped: SpeakerGroups = {}

    for audio_path, json_path in match_audio_and_metadata(real_dir):
        speaker_id = read_speaker_id(json_path, speaker_id_paths)
        grouped.setdefault(speaker_id, []).append(
            {
                "audio_path": audio_path,
                "json_path": json_path,
                "speaker_id": speaker_id,
            }
        )

    for samples in grouped.values():
        samples.sort(key=lambda sample: Path(sample["audio_path"]).as_posix())

    return dict(sorted(grouped.items()))


def _calculate_split_counts(
    speaker_count: int,
    ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    """비율에 가깝게 나누면서 각 분할에 최소 화자 한 명을 배정한다."""
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("train/validation/test 비율은 모두 0보다 커야 합니다.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train/validation/test 비율의 합은 1이어야 합니다.")
    if speaker_count < 3:
        raise ValueError(
            "train/validation/test를 만들려면 서로 다른 화자가 최소 3명 필요합니다."
        )

    exact_counts = [speaker_count * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact_counts]
    remainder = speaker_count - sum(counts)
    remainder_order = sorted(
        range(3),
        key=lambda index: exact_counts[index] - counts[index],
        reverse=True,
    )

    for index in remainder_order[:remainder]:
        counts[index] += 1

    for empty_index, count in enumerate(counts):
        if count > 0:
            continue
        donor_index = max(range(3), key=lambda index: counts[index])
        counts[donor_index] -= 1
        counts[empty_index] += 1

    return counts[0], counts[1], counts[2]


def split_grouped_speakers(
    grouped: SpeakerGroups,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> SpeakerSplit:
    """같은 화자가 서로 다른 분할에 들어가지 않도록 나눈다."""
    speaker_ids = sorted(grouped)
    random.Random(seed).shuffle(speaker_ids)

    train_count, validation_count, _ = _calculate_split_counts(
        len(speaker_ids),
        (train_ratio, validation_ratio, test_ratio),
    )

    validation_start = train_count
    test_start = train_count + validation_count
    split_speakers = {
        "train": speaker_ids[:validation_start],
        "validation": speaker_ids[validation_start:test_start],
        "test": speaker_ids[test_start:],
    }

    return {
        split_name: [
            sample
            for speaker_id in selected_speakers
            for sample in grouped[speaker_id]
        ]
        for split_name, selected_speakers in split_speakers.items()
    }


def create_speaker_split(
    real_dir: str | Path,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    speaker_id_paths: tuple[tuple[str, ...], ...] = DEFAULT_SPEAKER_ID_PATHS,
) -> SpeakerSplit:
    """Real 경로 하나로 화자별 그룹화와 세 데이터 분할을 수행한다."""
    grouped = group_audio_by_speaker(real_dir, speaker_id_paths)
    return split_grouped_speakers(
        grouped,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
