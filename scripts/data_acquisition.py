import os
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp

# import holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def retrieve_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros(4*33)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros(3*468)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros(3*21)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros(3*21)
    return np.concatenate([pose.flatten(), face.flatten(), lh.flatten(), rh.flatten()])


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return image


# set up webcam
cap = cv2.VideoCapture('http://192.168.1.122:8080/video')
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         frame, results = mediapipe_detection(frame, holistic)
#
#         landmarks = retrieve_keypoints(results)
#
#         drawn_image = draw_landmarks(frame, results)
#
#         cv2.imshow('OpenCV feed', cv2.flip(drawn_image, 1))
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()


num_sequences = 30
sequence_length = 60

DATA_PATH = '../data'
gestures = ['a_lot', 'dont_care', 'what']
for gesture in gestures:
    folder = os.path.join(DATA_PATH, gesture)
    os.makedirs(folder, exist_ok=True)
    num_existing_data = len(os.listdir(folder))
    for i in range(num_sequences):
        id = i + num_existing_data

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for gesture in gestures:
        folder = os.path.join(DATA_PATH, gesture)
        os.makedirs(folder, exist_ok=True)
        num_existing_data = len(os.listdir(folder))
        for i in range(num_sequences):
            id = i + num_existing_data
            for frame_num in range(sequence_length):

                ret, frame = cap.read()

                frame, results = mediapipe_detection(frame, holistic)
                landmarks = retrieve_keypoints(results)
                drawn_image = draw_landmarks(frame, results)

                # save the landmarks in a file!

                drawn_image = cv2.flip(drawn_image, 1)
                if frame_num == 0:
                    cv2.putText(drawn_image, 'STARTING COLLECTION', (120, 200),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(drawn_image, f'Collecting frame for {gesture}; video {i}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV feed', cv2.resize(drawn_image, (drawn_image.shape[1]//2, drawn_image.shape[0]//2)))
                    cv2.waitKey(2000)
                else:
                    cv2.putText(drawn_image, f'Collecting frame for {gesture}; video {i}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV feed', cv2.resize(drawn_image, (drawn_image.shape[1]//2, drawn_image.shape[0]//2)))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()

