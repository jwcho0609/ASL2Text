import cv2
import numpy as np
import os

def capture_frames(name, skip_frame):
    cap = cv2.VideoCapture(name)
    name_no_ext = os.path.splitext(name)
    folder_path = 'test' + "/" + name_no_ext[0]
    name_path = folder_path + "/" + name_no_ext[0]
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except OSError:
        print('Error: Creating directory of data')
    curr_frame = 0
    # Capture frame-by-frame
    ret, frame = cap.read()
    count = 0
    while ret:
        ret, frame = cap.read()
        if ret and curr_frame % skip_frame == 0:
            name = './' + name_path + str(count) + '.jpg'
            print('Creating...' + name)
            width = 224
            height = 224
            dim = (width, height)
            cropped = frame[300:1380, 0:1080]
            resized = cv2.resize(cropped, dim, interpolation= cv2.INTER_AREA)
            cv2.imwrite(name, resized)
            count += 1
        # To stop duplicate images
        curr_frame += 1
        if count == 30: break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

files = np.array(['A'])
for name in files:
    skip = 10
    filename = name + ".mp4"
    capture_frames(filename, skip)



