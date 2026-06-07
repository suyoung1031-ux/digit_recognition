"""
손글씨 숫자 인식 웹 애플리케이션
- MNIST 데이터셋으로 모델을 학습시키고
- 브라우저 캔버스에 그린 숫자를 인식합니다.
"""

import base64
import io
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import center_of_mass, shift, gaussian_filter
import pickle
import os

app = Flask(__name__)

model = None
scaler = None
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"


def 모델학습():
    global model, scaler

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("저장된 모델을 불러옵니다...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("모델 로드 완료!")
        return

    print("MNIST 데이터를 다운로드합니다... (처음 실행 시 시간이 걸립니다)")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target.astype(int)

    print("데이터 전처리 중...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("모델 학습 중... (2~3분 소요)")
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        max_iter=100,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True,
    )
    model.fit(X_scaled, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print("모델 학습 및 저장 완료!")


def 이미지전처리(이미지_데이터):
    """
    브라우저 캔버스 이미지를 MNIST 표준 형식(28×28, 무게중심 중앙 정렬)으로 변환합니다.
    """
    헤더, 인코딩된데이터 = 이미지_데이터.split(",", 1)
    이미지_바이트 = base64.b64decode(인코딩된데이터)

    이미지 = Image.open(io.BytesIO(이미지_바이트)).convert("L")
    이미지 = ImageOps.invert(이미지)
    픽셀 = np.array(이미지)

    행_존재 = np.any(픽셀 > 20, axis=1)
    열_존재 = np.any(픽셀 > 20, axis=0)

    if not np.any(행_존재):
        return np.zeros((1, 784))

    상단 = np.argmax(행_존재)
    하단 = len(행_존재) - np.argmax(행_존재[::-1])
    좌측 = np.argmax(열_존재)
    우측 = len(열_존재) - np.argmax(열_존재[::-1])
    크롭 = Image.fromarray(픽셀[상단:하단, 좌측:우측])

    # 종횡비 유지, 짧은 쪽 최소 3px 보장
    h, w = np.array(크롭).shape
    scale = 20.0 / max(h, w)
    new_h = max(3, int(h * scale))
    new_w = max(3, int(w * scale))
    크롭 = 크롭.resize((new_w, new_h), Image.LANCZOS)

    박스 = Image.new("L", (20, 20), 0)
    박스.paste(크롭, ((20 - new_w) // 2, (20 - new_h) // 2))

    결과이미지 = Image.new("L", (28, 28), 0)
    결과이미지.paste(박스, (4, 4))
    픽셀28 = np.array(결과이미지, dtype=np.float32)

    # 무게중심을 (14, 14)로 이동 (MNIST 표준)
    cy, cx = center_of_mass(픽셀28)
    픽셀28 = shift(픽셀28, [14 - cy, 14 - cx], cval=0)
    픽셀28 = np.clip(픽셀28, 0, 255)

    # 가우시안 블러로 획 확산
    픽셀28 = gaussian_filter(픽셀28, sigma=0.8)
    픽셀28 = np.clip(픽셀28, 0, 255)

    return 픽셀28.reshape(1, -1)


@app.route("/")
def 메인페이지():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def 숫자인식():
    데이터 = request.get_json()
    이미지_데이터 = 데이터.get("image")

    if not 이미지_데이터:
        return jsonify({"error": "이미지 데이터가 없습니다."}), 400

    픽셀_배열 = 이미지전처리(이미지_데이터)
    픽셀_배열_정규화 = scaler.transform(픽셀_배열)
    예측값 = model.predict(픽셀_배열_정규화)[0]
    확률_배열 = model.predict_proba(픽셀_배열_정규화)[0]
    확률_딕셔너리 = {str(i): round(float(확률_배열[i]) * 100, 1) for i in range(10)}

    return jsonify({
        "prediction": int(예측값),
        "probabilities": 확률_딕셔너리,
        "confidence": round(float(max(확률_배열)) * 100, 1)
    })


if __name__ == "__main__":
    모델학습()
    port = int(os.environ.get("PORT", 5001))
    print(f"\n서버 시작! 브라우저에서 http://localhost:{port} 을 열어주세요.\n")
    app.run(debug=False, host="0.0.0.0", port=port)
