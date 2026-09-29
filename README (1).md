# 딥보이스 탐지 - 독립 실행 버전 



## 폴더 구조 (전부 이 안에 있어야 함)

```
project/
├── README.md
├── configs/
│   ├── preprocess.yaml
│   └── model.yaml
├── src/
│   ├── deepvoice_diffusion/       # 전처리 담당
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── audio.py
│   │   └── dataset.py
│   ├── model.py                    # U-Net (노이즈 예측 모델)
│   └── diffusion.py                 # diffusion 프로세스 (잡음/loss/점수계산)
├── scripts/
│   ├── preprocess.py                 # ① Real wav → log-mel 변환 (학습용)
│   ├── train.py                      # ② 학습
│   ├── score.py                      # ③ 음성 하나 점수 계산
│   └── find_threshold.py             # ④ threshold/EER + 분포 그래프
└── tests/
    └── test_model.py                  # 모델 자체 검증 (데이터 없어도 실행 가능)
```

`data/`, `checkpoints/`, `artifacts/` 폴더는 스크립트를 실행하면 자동으로 생깁니다.

## 처음 한 번만 하는 준비

```powershell
cd 이 폴더 경로
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install librosa matplotlib numpy pyyaml soundfile tqdm torch scikit-learn
```

## 데이터 준비 (중요)



- **학습용 Real**
- **검증용 Real**
- **검증용 Fake**

## 실행 순서

매번 새 터미널을 열 때는 `.\.venv\Scripts\Activate.ps1`로 `(.venv)`부터 켤 것.

1. **`configs/preprocess.yaml`** 열어서 `dataset.input_dir`을 **학습용** Real wav 폴더 경로로 수정
2. 전처리:
   ```powershell
   python scripts\preprocess.py
   ```
3. (선택) 모델 코드 자체 검증 — 데이터 없이도 실행 가능:
   ```powershell
   pytest tests\test_model.py
   ```
4. 학습:
   ```powershell
   python scripts\train.py
   ```
   빠르게 맛보기만 하려면 `python scripts\train.py --max-batches 5` 처럼 실행.
5. 성능 확인 + 분포 그래프 저장 (검증용 Real 폴더, 검증용 Fake 폴더 경로를 각각 넣기):
   ```powershell
   python scripts\find_threshold.py --real-dir "검증용 Real 폴더 경로" --fake-dir "검증용 Fake 폴더 경로" --checkpoint checkpoints\best_model.pt --preprocess-config configs\preprocess.yaml --eval-timesteps 20 60 100 150
   ```

## configs/model.yaml 기본값

지금은 CPU에서 빠르게 확인만 하도록 축소된 값(epochs=5, base_channels=16 등)으로
되어 있습니다. 제대로 학습시키려면 파일 맨 아래 "정식 규모로 키울 때" 주석을 참고해서
값을 늘리고 4번부터 다시 실행하면 됩니다. (`--eval-timesteps`도 `diffusion.timesteps`보다
작은 값들로 같이 맞춰줘야 합니다.)

## 참고

검증용 Real을 다른 출처에서 가져올 때는 가능하면 학습용 Real과 녹음 환경(마이크,
배경 소음 유무 등)이 비슷한 것을 고르는 게 좋습니다. 환경 차이가 너무 크면
"Fake라서 이상하다"가 아니라 "녹음 환경이 달라서 이상하다"를 감지하는 것일 수 있습니다.
