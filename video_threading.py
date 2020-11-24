from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import threading

test = 0

def video():
    # 모델 불러오기
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    test = 1

    # video stream 시작
    vs = VideoStream(0).start()
    time.sleep(2.0)

    test = 2

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        cv2.imshow("webcam", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            print(1)
            test = 1
        if key == ord("2"):
            test = 2

        if key == ord("q"):
            break

    # 종료
    cv2.destroyAllWindows()
    vs.stop()

videoThread = threading.Thread(target=video)
videoThread.start()

preT = -1

while True:
    print(test)
