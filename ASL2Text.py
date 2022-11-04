import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

alpha = ["A", "B", "C", "D", "Del", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "Nothing", "O",
         "P", "Q", "R", "S", "Space", "T", "U", "V", "W", "X", "Y", "Z"]
models = []
array = []
global word
word = ""
global previous_char
previous_char = "Nothing"


def image_2_char(image):
    global word
    global previous_char
    image = image / 255
    prediction = np.zeros(29)
    for model in models:
        prediction += model.predict(np.array([image]))[0]
    test_predictions = np.argmax(prediction)
    array.append(prediction)
    char = alpha[test_predictions]
    if len(array) > 5:
        array.pop(0)
        array2 = np.array(array)
        array2 = np.sum(array2, axis=0)
        highest_count = max(array2)
        highest_percent = highest_count / len(array)
        if highest_percent >= 0.9:
            current_char = alpha[np.argmax(array2)]
            if previous_char != current_char:
                if current_char != "Nothing":
                    if current_char == "Del":
                        word = word[0: -1]
                    elif current_char == "Space":
                        word += " "
                    else:
                        word += current_char
                print(current_char)
                previous_char = current_char
    print(word + "     current character:" + char)


def start(skip_frame=5):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    curr_frame = 0
    while ret:
        ret, frame = cap.read()
        if ret and curr_frame % skip_frame == 0:
            dim = (64, 64)
            cropped = frame[80:560, 0:480]
            resized = cv2.resize(cropped, dim)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            image_2_char(resized)
            cv2.imshow("Video", cropped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        curr_frame += 1
    cap.release()
    cv2.destroyAllWindows()


def load_models(names):
    for name in names:
        models.append(load_model("models/" + name + ".h5"))


def main():
    load_models(["cnn64ddT9_99_82"])
    start()


if __name__ == '__main__':
    main()
