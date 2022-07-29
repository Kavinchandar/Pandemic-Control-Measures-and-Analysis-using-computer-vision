# imports
import time
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

# keras model imports
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# facemodel
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
facemodel = cv2.dnn.readNet(prototxtPath, weightsPath)
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# mask model
maskmodel = load_model("mask_detector.model")

# test


def test():
    New_Colors = ['green', 'blue', 'purple', 'brown', 'teal']
    # bar graph
    plt.bar(avg_fps, test_time, color=New_Colors)
    plt.title('Face-mask recognition system')
    plt.xlabel('FPS')
    plt.ylabel('Time (s)')
    plt.show()
    # line graph
    plt.plot(test_time, avg_fps)
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.show()

# fps counter


def fps(picture, prev_Time):
    current_Time = time.time()
    fps = 1 / (current_Time - prev_Time)

    avg_fps.append(fps)
    test_time.append(current_Time)
    prev_Time = current_Time
    cv2.putText(picture, f'FPS: {int(fps)}', (300, 400),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    return prev_Time


# analysis
start = 0
end = 0


def duration():
    global end
    end = time.time()
    dur = end - start
    return dur


def formatter(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))


def analysis():
    label = ['Mask on', 'Mask worn incorrectly', 'Mask off']
    total = m_on+m_half+m_off
    if total != 0:
        percent = [f'{int((m_on/total)*100)}%',
                   f'{int((m_half/total)*100)}%', f'{int((m_off/total)*100)}%']
        colours = ['green', 'cyan', 'red']
        myexplode = [0.02, 0.01, 0.01]
        dur = duration()
        plt.pie(np.array([m_on, m_half, m_off]), labels=percent,
                explode=myexplode, shadow=False, colors=colours, radius=0.6)
        plt.legend(title='Status Frequency', loc='lower right', labels=label,
                   bbox_to_anchor=(1.3, 0.8))
        plt.text(-1.5, 1, f'Performance: {int(sum(avg_fps)/len(avg_fps))} fps',
                 family='serif', ha='left')
        plt.text(-1.5, -0.9, f'Total Duration: {formatter(dur)}',
                 family='serif', ha='left')
        plt.text(-1.5, -1.0, f'Mask on: {formatter(dur*(int(percent[0][:-1])/100))}',
                 family='serif', ha='left')
        plt.text(-1.5, -1.2, f'Incorrectly worn: {formatter(dur*(int(percent[1][:-1])/100))}',
                 family='serif', ha='left')
        plt.text(-1.5, -1.1, f'Mask off: {formatter(dur*(int(percent[2][:-1])/100))}',
                 family='serif', ha='left')
        plt.title('Results')
        plt.show()
    else:
        ax = plt.gca()
        plt.text(0.5, 0.5, 'No Face Detected', family='serif', ha='center')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.title('Results')
        plt.show()


# mask prediction


def mask_prediction(frame, facemodel, maskmodel):
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))
    facemodel.setInput(blob)
    detections = facemodel.forward()
    faces, locs, preds = [], [], []
    for i in range(0, detections.shape[2]):
        confidence = detections[0][0][i][2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * \
                np.array([width, height, width, height])
            (x1, y1, x2, y2) = box.astype("int")
            (x1, y1) = (max(0, x1), max(0, y1))
            (x2, y2) = (min(width - 1, x2), min(height - 1, y2))
            face = frame[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((x1, y1, x2, y2))
    if faces:
        faces = np.array(faces, dtype="float32")
        preds = maskmodel.predict(faces, batch_size=32)
    return (locs, preds)

# nose detection


def nose_detection(frame, landmarks, color, draw):
    if landmarks.part(30) and landmarks.part(31) and landmarks.part(35) and landmarks.part(33):
        nose_top_x, nose_top_y = landmarks.part(30).x, landmarks.part(30).y
        nose_left_x, nose_left_y = landmarks.part(31).x, landmarks.part(31).y
        nose_right_x, nose_right_y = landmarks.part(35).x, landmarks.part(35).y
        nose_bottom_x, nose_bottom_y = landmarks.part(
            33).x, landmarks.part(33).y
        if draw:
            cv2.circle(frame, (nose_top_x, nose_top_y), 3, color, -1)
            cv2.circle(frame, (nose_left_x, nose_left_y), 3, color, -1)
            cv2.circle(frame, (nose_right_x, nose_right_y), 3, color, -1)
            cv2.circle(frame, (nose_bottom_x, nose_bottom_y), 3, color, -1)
        return True
    return False

# mouth detection


def mouth_detection(frame, landmarks, color, draw):
    if landmarks.part(51) and landmarks.part(48) and landmarks.part(54) and landmarks.part(57):
        mouth_top_x, mouth_top_y = landmarks.part(51).x, landmarks.part(51).y
        mouth_left_x, mouth_left_y = landmarks.part(48).x, landmarks.part(48).y
        mouth_right_x, mouth_right_y = landmarks.part(
            54).x, landmarks.part(54).y
        mouth_bottom_x, mouth_bottom_y = landmarks.part(
            57).x, landmarks.part(57).y
        if draw:
            cv2.circle(frame, (mouth_top_x, mouth_top_y), 3, color, -1)
            cv2.circle(frame, (mouth_left_x, mouth_left_y), 3, color, -1)
            cv2.circle(frame, (mouth_right_x, mouth_right_y), 3, color, -1)
            cv2.circle(frame, (mouth_bottom_x, mouth_bottom_y), 3, color, -1)
        return True
    return False

# eye detection


def eye_detection(frame, landmarks, color, draw):
    if landmarks.part(36) and landmarks.part(39) and landmarks.part(42) and landmarks.part(45):
        lefteye_left_x, lefteye_left_y = landmarks.part(
            36).x, landmarks.part(36).y
        lefteye_right_x, lefteye_right_y = landmarks.part(
            39).x, landmarks.part(39).y
        righteye_left_x, righteye_left_y = landmarks.part(
            42).x, landmarks.part(42).y
        righteye_right_x, righteye_right_y = landmarks.part(
            45).x, landmarks.part(45).y
        if draw:
            cv2.circle(frame, (lefteye_left_x, lefteye_left_y), 3, color, -1)
            cv2.circle(frame, (lefteye_right_x, lefteye_right_y), 3, color, -1)
            cv2.circle(frame, (righteye_left_x, righteye_left_y), 3, color, -1)
            cv2.circle(frame, (righteye_right_x,
                               righteye_right_y), 3, color, -1)
        return True
    return False

# mask status function


m_on, m_off, m_half = 0, 0, 0
avg_fps = []
test_time = []


def status(mask, withoutMask, nose, mouth, eye):
    global m_on
    global m_half
    global m_off
    if mask > withoutMask and not nose and not mouth:
        label = "Mask on"
        color = (0, 255, 0)
        m_on += 1
    elif (nose or mouth) and withoutMask < 0.9999:
        label = "Mask worn incorrectly"
        color = (255, 255, 0)
        m_half += 1
    else:
        label = "Mask off"
        color = (0, 0, 255)
        m_off += 1
    return (label, color)


# main function
def main():
    global start
    print("Starting Face-mask Detection protocol...")
    capture = cv2.VideoCapture(0)
    prev_Time = 0
    start = time.time()
    while True:
        nose, mouth, eye = False, False, False

        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        locs, preds = mask_prediction(frame, facemodel, maskmodel)
        faces = face_detector(frame)

        for face in faces:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            landmarks = predictor(gray, face)
            nose = nose_detection(frame, landmarks, (255, 0, 255), draw=True)
            mouth = mouth_detection(frame, landmarks, (255, 255, 0), draw=True)
            eye = eye_detection(frame, landmarks, (255, 0, 0), draw=True)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            #print(mask*100, withoutMask*100)
            label, color = status(mask, withoutMask, nose, mouth, eye)
            cv2.putText(frame, label, (startX, startY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            confidence = str(max(mask, withoutMask) * 100)[:4]+'%'
            cv2.putText(frame, confidence, (startX, endY + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        prev_Time = fps(frame, prev_Time)
        cv2.imshow("Face-mask recognition system", frame)
        if cv2.waitKey(10) & 0xFF == ord('x'):
            analysis()
            test()
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
