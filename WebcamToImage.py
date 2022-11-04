import cv2
from tensorflow.keras.models import load_model
import numpy as np

alpha = ["A", "B", "C", "D", "Del", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "Nothing", "O",
         "P", "Q", "R", "S", "Space", "T", "U", "V", "W", "X", "Y", "Z"]
models = []


def predict(image):
    image = image / 255
    prediction = np.zeros(29)
    for model in models:
        prediction += model.predict(np.array([image]))[0]
    test_predictions = np.argmax(prediction)
    print(alpha[test_predictions])


def webcam_to_images(skip_frame=5):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    count = 0
    curr_frame = 0
    while ret:
        ret, frame = cap.read()
        if ret and curr_frame % skip_frame == 0:
            width = 64
            height = 64
            dim = (width, height)
            cropped = frame[80:560, 0:480]
            resized = cv2.resize(cropped, dim)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            predict(resized)
            cv2.imshow("Video", cropped)
            # plt.imshow(resized)
            # plt.show()
            count += 1
        # To stop duplicate images
        curr_frame += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def load_models(names):
    for name in names:
        models.append(load_model("models/" + name + ".h5"))


def main():
    load_models(["cnn64ddT9_99_82"])
    webcam_to_images()


if __name__ == '__main__':
    main()
