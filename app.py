import cv2
from flask import Flask, render_template, Response
from camera import Camera
import pickle

app = Flask(__name__)

# XGBoost 모델 불러오기
#with open("my_model.pkl", "rb") as f:
#    model = pickle.load(f)

#posture_label = model.predict()

@app.route('/')
def index():
    return render_template('index.html')

def annoate_frame(frame, posture_label):
    # 0: normal (초록) 1: tech_neck(노랑) 2: reclining (주황)
    colors = [ (0, 255, 0),(255, 255, 0), (255, 165, 0)]

    if posture_label == 0:
        text = "normal"
        color = colors[0]
    elif posture_label == 1:
        text = "tech_neck"
        color = colors[1]
    elif posture_label == 2:
        text = "Reclining"
        color = colors[2]
    else:
        text = "Unknown"
        color = (255, 255, 255)  # 흰색

    # 이미지에 텍스트 추가
    annotated_frame = frame.copy()
    cv2.putText(annotated_frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return annotated_frame


def gen(camera):
    while True:
        frame = camera.get_frame()

        # 이미지를 특성 벡터로 변환
        feature_vector = preprocess_image(frame)

        # 훈련된 모델(XGBoost)을 사용하여 자세 분석
        posture_label = model.predict(feature_vector)

        # 분석 결과를 이미지에 표시
        annotated_frame = annotate_frame(frame, posture_label)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + annotated_frameframe + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary-frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0')




