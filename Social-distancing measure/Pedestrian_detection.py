from cProfile import label
import numpy as np
import math
import cv2
import imutils
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import time

NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2
MIN_DISTANCE = 40


def distance_function(x1, y1, x2, y2):
    return ((x2 - x1)**2 - (y2-y1)**2)**(0.5)


# test


def test():
    New_Colors = ['green', 'blue', 'purple', 'brown', 'teal']
    # bar graph
    plt.bar(avg_fps, test_time, color=New_Colors)
    plt.title('Social-distancing measure system')
    plt.xlabel('FPS')
    plt.ylabel('Time (s)')
    plt.show()
    # line graph
    plt.plot(test_time, avg_fps)
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.show()

# fps counter


prev_time = 0
avg_fps = []
test_time = []


def fps(picture, prev_Time):
    current_Time = time.time()
    fps = 1 / (current_Time - prev_Time)
    avg_fps.append(fps)
    test_time.append(current_Time)
    prev_Time = current_Time
    cv2.putText(picture, f'FPS: {int(fps)}', (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return prev_Time
# analysis code


right, wrong = 0, 0


def analysis(right, wrong):
    total = right + wrong
    right_percent = (right / total)*100
    wrong_percent = 100 - right_percent
    label = [str(right_percent)[:2]+'%',
             str(wrong_percent)[:2]+'%']
    my_explode = [0, 0.2]
    plt.pie(np.array([right, wrong]), labels=label,
            explode=my_explode, shadow=False, radius=0.8)
    plt.legend(title="Frequency", labels=["Conformity", "Deviance"], loc='lower right',
               bbox_to_anchor=(1.3, 0.8))
    plt.text(-2, -0.6, f'Performance: {int(sum(avg_fps)/len(avg_fps))} fps')
    plt.text(-2, -0.7, f'Total People: {total}')
    plt.text(-2, -0.8, f'Followers: {right}')
    plt.text(-2, -0.9, f'Deviants: {wrong}')
    plt.title('Results')
    plt.show()

# pedestrian detection


def pedestrian_detection(image, model, layer_name, personidz=0):
    (H, W) = image.shape[:2]
    results = []
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)
    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personidz and confidence > MIN_CONFIDENCE:

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)

    if len(idzs) > 0:
        for i in idzs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            res = [confidences[i], (x, y, x + w, y + h), centroids[i]]
            results.append(res)

    return results

# centroid dist visualization


def centroid_visual(image, centroids, draw=False):
    if draw:
        n = len(centroids)
        for i in range(0, n):
            for j in range(i+1, n):
                if distance_function(centroids[i][0], centroids[i][1], centroids[j][0], centroids[j][1]) < MIN_DISTANCE:
                    cv2.line(image, centroids[i],
                             centroids[j], (255, 255, 0), 2)
    else:
        pass


def violations(image, right, wrong):
    violation = set()
    total = len(results)

    if total >= 2:
        centroids = np.array([res[2] for res in results])
        n = len(centroids)
        centroid_visual(image, centroids, True)
        distance = dist.cdist(centroids, centroids, metric='euclidean')
        for i in range(0, distance.shape[0]):
            for j in range(i+1, distance.shape[1]):
                if distance[i][j] < MIN_DISTANCE:
                    violation.add(i)
                    violation.add(j)

    for (i, (prob, bbox, centroid)) in enumerate(results):
        (x1, y1, x2, y2) = bbox
        (cx, cy) = centroid
        color = (0, 255, 0)
        if i in violation:
            color = (0, 0, 255)
            wrong += 1
        else:
            right += 1
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      color, 2)
        cv2.circle(image, (cx, cy), 3, color, 2)

    cv2.putText(image, f'Total: {total}', (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(image, f'Followers: {right}', (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(image, f'Deviants: {wrong} ', (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image, right, wrong


labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"
model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
'''
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
'''

layer_name = model.getLayerNames()
layer_name = [layer_name[i[0] - 1] for i in model.getUnconnectedOutLayers()]

cap = cv2.VideoCapture("top down view.mp4")
#cap = cv2.VideoCapture("part1.mp4")
#cap = cv2.VideoCapture("part 2.mp4")


while cap.isOpened():
    _, frame = cap.read()
    frame = imutils.resize(frame, width=700)
    results = pedestrian_detection(frame, model, layer_name,
                                   personidz=LABELS.index("person"))
    frame, right, wrong = violations(frame, 0, 0)
    prev_time = fps(frame, prev_time)
    cv2.imshow("Social Distance Measure", frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        analysis(right, wrong)
        test()
        break


cap.release()
cv2.destroyAllWindows()
