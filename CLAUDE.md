# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

Flask 기반 손글씨 숫자 인식 웹 앱. 브라우저 캔버스에서 숫자를 그리면 서버가 ML 모델로 예측해 결과를 반환한다.

## 실행 명령어

```bash
# WSL 터미널에서
cd "/mnt/c/Users/연우사랑/Desktop/클로드코드 바이브코딩/Study-01/web_version"
PATH="$HOME/.local/bin:$PATH" python3 app.py    # http://localhost:5000
```

**처음 실행 시** `model.pkl` / `scaler.pkl`이 없으면 MNIST 데이터 다운로드 후 자동 학습 (1~2분).  
이후 실행부터는 저장된 파일을 즉시 불러온다.

### 의존성 설치

```bash
~/.local/bin/pip install flask scikit-learn numpy pillow pandas --break-system-packages
```

> pip 실행 파일 위치: `~/.local/bin/pip` (시스템 pip와 별도)

## 아키텍처

### 파일 구조

```
web_version/
├── app.py              # Flask 서버 + 모델 학습/추론 전체
├── templates/
│   └── index.html      # 캔버스 UI + fetch API 인식 요청
├── model.pkl           # 학습된 MLPClassifier (gitignore 권장)
└── scaler.pkl          # StandardScaler (model.pkl과 쌍으로 관리)
```

### 요청 흐름

```
브라우저 캔버스 → (임시 캔버스에 흰 배경 합성) → base64 PNG
  → POST /predict → 이미지전처리() → scaler.transform() → model.predict()
  → JSON { prediction, confidence, probabilities } → 확률 막대 렌더링
```

### 이미지 전처리 (핵심)

캔버스 PNG(흰 배경·검은 글씨) → 흑백 변환 → 색상 반전 → 글씨 영역 크롭 → 20×20 리사이즈 → 28×28 중앙 배치  
MNIST 학습 데이터와 동일한 포맷을 맞추는 것이 인식률의 핵심이다.

### 모델

- `MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', max_iter=20)`
- MNIST 6만 장 학습, 테스트 정확도 **97.35%**
- `StandardScaler`로 픽셀 정규화 후 입력
