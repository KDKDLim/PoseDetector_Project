# 라즈베리 파이에서 영상 캡쳐 및 전송

import cv2
import requests

# 서버 URL
SERVER_URL = 'http://192.168.50.245:5000'

# 카메라 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # 프레임 캡쳐

    # 프레임을 JPEG 형식으로 인코딩
    _, img_encoded = cv2.imencode('.jpg', frame)

    # 이미지 데이터를 바이트로 변환
    image_bytes = img_encoded.tobytes()

    # 서버로 POST 요청 전송
    reponse = requests.post(SERVER_URL, data=image_bytes)

    # 서버 응답 확인
    if response.status.code == 200:
        print('Frame sent successfully')

cap.release()  # 카메라 해제