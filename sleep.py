from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def get_ear(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

# 모델 불러오기
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# video stream 시작
print("[INFO] camera sensor warming up...")
vs = VideoStream(0).start()
time.sleep(2.0)

# blink 관련

STATE = "normal"
COUNTER = 0
TOTAL = 0

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 300

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print(lStart, lEnd)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = get_ear(leftEye)
        rightEAR = get_ear(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # 눈 윤곽 표시
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #졸음 감지


        if ear < EYE_AR_THRESH: # 눈을 감았을 때
            COUNTER += 1

            # 눈을 계속 감고 있는 경우 -> 졸음이라고 판단
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                STATE = "sleep!"

        else:   # 눈을 떴을 때
            COUNTER = 0
            STATE = "normal"

        cv2.putText(frame, "COUNTER: {}".format(COUNTER), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "USER STATE: {}".format(STATE), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("webcam", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# 종료
cv2.destroyAllWindows()
vs.stop()