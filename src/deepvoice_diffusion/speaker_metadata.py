import json
from pathlib import Path


# 프로젝트 최상위 폴더에서 실행할 때 사용할 원본 데이터 경로
RAW_FOLDER = Path("data/raw")

# 화자별 WAV 파일 목록을 저장할 빈 사전
speaker_wavs = {}


def organize_wavs(audio_folder):    #organize_wavs: 음성 파일 경로
    # WAV 파일 이름과 실제 파일 경로를 저장한다.
    wav_by_name = {}

    for wav_file in audio_folder.rglob("*.wav"):
        wav_name = wav_file.stem

        # 같은 이름이 이미 저장되어 있으면 두 경로를 출력하고 중단한다.
        if wav_name in wav_by_name:
            first_file = wav_by_name[wav_name]

            raise ValueError(
                f"중복된 WAV 파일명입니다.\n"
                f"첫 번째 파일: {first_file}\n"
                f"두 번째 파일: {wav_file}"
            )

        wav_by_name[wav_name] = wav_file

    return wav_by_name


def read_speaker_id(dataset_name, json_file):   ## read_speaker_id: 화자 ID 확인
    # JSON 파일을 열어 파이썬 사전으로 읽는다.
    with open(json_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    # 자유대화 데이터에서 화자 ID를 읽는다.
    if dataset_name == "free_conversation":
        return str(json_data["녹음자정보"]["recorderId"])

    # 다화자 음성합성 데이터에서 화자 ID를 읽는다.
    if dataset_name == "multispeaker_tts":
        return str(json_data["기본정보"]["NumberOfSpeaker"])

    # 위의 두 데이터셋이 아니라면 화자 ID 위치를 알 수 없으므로 중단한다.
    raise ValueError(f"화자 ID 형식을 모르는 데이터셋입니다: {dataset_name}")


def collect_dataset(dataset_name):
    # 하나의 데이터셋에서 audio 폴더와 metadata 폴더의 경로를 만든다.
    dataset_folder = RAW_FOLDER / dataset_name
    audio_folder = dataset_folder / "audio"
    metadata_folder = dataset_folder / "metadata"

    # 필요한 폴더가 실제로 존재하는지 확인한다.
    if not audio_folder.is_dir() or not metadata_folder.is_dir():
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {dataset_folder}")

    # WAV 파일과 JSON 파일의 목록을 가져온다.
    wav_by_name = organize_wavs(audio_folder)
    json_files = list(metadata_folder.rglob("*.json"))

    # JSON마다 화자 ID를 읽고 같은 이름의 WAV를 찾는다.
    for json_file in json_files:
        speaker_id = read_speaker_id(dataset_name, json_file)
        speaker_name = dataset_name + "/" + speaker_id
        wav_file = wav_by_name[json_file.stem]

        # 처음 발견한 화자라면 빈 WAV 목록을 만든다.
        if speaker_name not in speaker_wavs:
            speaker_wavs[speaker_name] = []

        # 찾은 WAV 파일을 해당 화자의 목록에 추가한다.
        speaker_wavs[speaker_name].append(wav_file)

    # 이 데이터셋에서 찾은 파일 수를 출력한다.
    print(f"{dataset_name}: WAV {len(wav_by_name)}개, JSON {len(json_files)}개")


# 두 실제 음성 데이터셋을 차례대로 처리한다.
collect_dataset("free_conversation")
collect_dataset("multispeaker_tts")


# 화자별로 묶인 WAV 파일 수를 출력한다.
print("\n화자별 WAV 파일 수")

total_wav_count = 0

for speaker_name, wav_files in sorted(speaker_wavs.items()):
    print(f"{speaker_name}: {len(wav_files)}개")
    total_wav_count += len(wav_files)

# 전체 화자 수와 WAV 파일 수를 출력한다.
print(f"\n전체 화자: {len(speaker_wavs)}명")
print(f"전체 WAV 파일: {total_wav_count}개")
