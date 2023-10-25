from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import cv2
import mediapipe as mp
import os
from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# Load the Keras model for sign language detection
model_path = os.path.abspath('E:\GitHubRepo\Sign-Language-Detection-Website\model.h5')
classifier = load_model(model_path)

# Load the MediaPipe model for hand gesture recognition
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Define the actions (letters) and colors
actions = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
           "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
           "U", "V", "W", "X", "Y", "Z"]
colors = [(245, 117, 16)] * len(actions)

# Define the threshold for action prediction
threshold = 0.8

# Initialize variables for hand gesture recognition
sequence = []
sentence = []
accuracy = []
predictions = []

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Make detections using MediaPipe
    crop_frame = frame[40:400, 0:300]
    frame = cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    image, results = mediapipe_detection(crop_frame, hands)
    
    # Draw landmarks
    # draw_styled_landmarks(image, results)
    
    # Extract keypoints
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    try:
        if len(sequence) == 30:
            res = classifier.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)] * 100))
                    else:
                        sentence.append(actions[np.argmax(res)])
                        accuracy.append(str(res[np.argmax(res)] * 100))

            if len(sentence) > 1:
                sentence = sentence[-1:]
                accuracy = accuracy[-1:]

            cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
            cv2.putText(frame, "Output: -" + ' '.join(sentence) + ''.join(accuracy), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    except Exception as e:
        pass

    cv2.imshow('Combined Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
