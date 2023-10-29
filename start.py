from flask import Flask, render_template, request, Response, send_from_directory
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")

def detect():
    actions = ['나', '슬프다', '즐겁다']
    seq_length = 30

    # Set the Korean font file path
    font_path = './font/NanumGothic.ttf'  # 경로를 실제 파일 위치로 변경

    # Load Korean fonts.
    font_size = 50
    font = ImageFont.truetype(font_path, font_size)
    font_color = (255, 255, 255)
    font_thickness = 6

    model = load_model('./models/model_r2.h5')

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    # cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Webcam", 200, 150)

    # Set the webcam's width and height
    cap.set(3, 400)  # Width to 1920 pixels
    cap.set(4, 300)  # Height to 1080 pixels
    # cv2.resizeWindow("Webcam", 800, 600)

    seq = []
    action_seq = []

    while cap.isOpened():
        ret, img = cap.read()

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                angle = np.degrees(angle)

                d = np.concatenate([joint.flatten(), angle])
                seq.append(d)
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0, 0), thickness=1,
                                                                                       circle_radius=0),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0, 0),
                                                                                         thickness=1, circle_radius=0))

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.95:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 1:
                    continue

                this_action = '?'
                if action_seq[-1]:
                    this_action = action

                # 한글 텍스트를 이미지로 렌더링
                if this_action in actions:
                    pil_img = Image.fromarray(img)
                    draw = ImageDraw.Draw(pil_img)
                    text_size = draw.textsize(this_action, font=font)
                    x = int(0.5 * img.shape[1])  # 가로 위치를 이미지 가로의 20% 지점으로 수정
                    y = int(0.8 * img.shape[0])  # 세로 위치를 이미지 세로의 80% 지점으로 수정
                    draw.text((x, y), this_action, font=font, fill=font_color)
                    img = np.array(pil_img)

        cv2.imshow('Gesture Recognition', img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


@app.route("/video_feed")
def video_feed():
    return Response(detect(), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == "__main__":
    app.run(debug=True)
