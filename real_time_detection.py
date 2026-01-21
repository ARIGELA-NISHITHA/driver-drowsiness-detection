import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
from playsound import playsound

# Load CNN models
eye_model = load_model('../models/eye_model.h5')
yawn_model = load_model('../models/yawn_model.h5')

# Facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')

LEFT_EYE = list(range(36,42))
RIGHT_EYE = list(range(42,48))
MOUTH = list(range(60,68))

DROWSY_FRAMES = 20
counter = 0
alarm_on = False

def crop_region(frame, points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return frame[y1:y2, x1:x2]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE])
        mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in MOUTH])

        # Crop and resize
        leye_crop = cv2.resize(crop_region(frame, left_eye), (64,64))/255.0
        reye_crop = cv2.resize(crop_region(frame, right_eye), (64,64))/255.0
        mouth_crop = cv2.resize(crop_region(frame, mouth), (64,64))/255.0

        leye_crop = np.expand_dims(leye_crop, axis=0)
        reye_crop = np.expand_dims(reye_crop, axis=0)
        mouth_crop = np.expand_dims(mouth_crop, axis=0)

        left_pred = eye_model.predict(leye_crop)[0][0]
        right_pred = eye_model.predict(reye_crop)[0][0]
        mouth_pred = yawn_model.predict(mouth_crop)[0][0]

        drowsy = (left_pred > 0.5 and right_pred > 0.5) or (mouth_pred > 0.5)

        if drowsy:
            counter += 1
            if counter >= DROWSY_FRAMES:
                if not alarm_on:
                    alarm_on = True
                    playsound('../alarm.wav')
                cv2.putText(frame, "DROWSINESS ALERT!", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            counter = 0
            alarm_on = False

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
