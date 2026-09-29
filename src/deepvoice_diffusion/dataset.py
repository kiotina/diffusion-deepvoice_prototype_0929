# 전처리할 파일 목록 관리

from __future__ import annotations

from pathlib import Path


def find_wav_files(input_dir: str | Path) -> list[Path]:
    """입력 폴더 아래에 있는 모든 WAV 파일을 재귀적으로 찾는다."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {input_dir}")

    # 정렬해 두면 운영체제가 파일을 반환하는 순서와 관계없이
    # 실행할 때마다 같은 index와 같은 출력 파일명을 사용할 수 있다.
    return sorted(
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"
    )


def select_files(files: list[Path], minimum: int, maximum: int | None) -> list[Path]:
    """최소 데이터 개수를 보장하고 설정된 최대 개수만큼 선택한다."""
    # 잘못된 경로나 불완전한 데이터로 학습이 시작되는 것을 미리 막는다.
    if len(files) < minimum:
        raise RuntimeError(
            f"Need at least {minimum:,} WAV files, but found only {len(files):,}."
        )

    # maximum이 None이면 현재 발견한 파일을 전부 사용한다.
    return files if maximum is None else files[:maximum]
