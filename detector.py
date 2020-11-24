from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import threading
import time
from queue import Queue
from scipy.spatial import distance as dist
from PyQt5.QtCore import *


class MyDetector:

    # sleep 관련
    STATE = "normal"
    COUNTER = 0
    TOTAL = 0

    EYE_AR_THRESH = 0.3
    SLEEP_CONSEC_FRAMES = 50

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def get_ear(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_sleep(self, shape, frame):
        leftEye = shape[self.lStart: self.lEnd]
        rightEye = shape[self.rStart:self.rEnd]
        leftEAR = self.get_ear(leftEye)
        rightEAR = self.get_ear(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # 눈 윤곽 표시
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 졸음 감지
        if ear < self.EYE_AR_THRESH:  # 눈을 감았을 때
            self.COUNTER += 1

            # 눈을 계속 감고 있는 경우 -> 졸음이라고 판단
            if self.COUNTER >= self.SLEEP_CONSEC_FRAMES:
                self.STATE = "sleep!"

        else:  # 눈을 떴을 때
            COUNTER = 0
            self.TATE = "normal"

        cv2.putText(frame, "COUNTER: {}".format(self.COUNTER), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "USER STATE: {}".format(self.STATE), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def video(self):



        # 모델 불러오기
        print('load model...')
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("public/shape_predictor_68_face_landmarks.dat")

        # video stream 시작
        print('start video stream')
        vs = VideoStream(0).start()
        time.sleep(2.0)

        print('start detecting')
        while True:

            frame = vs.read()
            frame = imutils.resize(frame, width=600)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                self.detect_sleep(shape, frame)

            cv2.imshow("webcam", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("1"):
                self.sec = 1
                self.sec_changed.emit('{}'.format(self.sec))
            if key == ord("2"):
                self.sec = 2
                self.sec_changed.emit('{}'.format(self.sec))
            if key == ord("3"):
                self.sec = 3
                self.sec_changed.emit('{}'.format(self.sec))
            if key == ord("4"):
                self.sec = 4
                self.sec_changed.emit('{}'.format(self.sec))
            if key == ord("5"):
                self.sec = 5
                self.sec_changed.emit('{}'.format(self.sec))
            if key == ord("6"):
                self.sec = 6
                self.sec_changed.emit('{}'.format(self.sec))

            if key == ord("q"):
                break



## __main__

if __name__=='__main__':
    q = Queue()
    # t = threading.Thread(target=MyDetector.video(q), args=(q,))
    # t.start()
    md = MyDetector()
    md.video(q)

