import cv2
import dlib
import numpy as np


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fps, frames = frame.get(cv2.CAP_PROP_FPS), frame.get(cv2.CAP_PROP_FRAME_COUNT)
    cv2.putText(frame, "Press SPACE: FOR EMOTION", (5, 470), fps, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0), 2)

        landmarks = predictor(gray, face)
        PT = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            if (n == 37) | (n == 40) | (n == 43) | (n == 46):
                PT = PT + [(x, y)]
                #print(PT)

                #cv2.putText(frame, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                #cv2.circle(frame,(x,y), 3, (255,0,0), -1)

        # Right eye
        cv2.rectangle(frame, PT[0], PT[1], (0, 255, 0), 1)

        # Left eye
        cv2.rectangle(frame, PT[2], PT[3], (0, 255, 0), 1)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break