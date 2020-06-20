# load the train model and test with live camera
import cv2
import numpy as np
from tensorflow.keras import models

new_model = models.load_model(r"Mask_on_Face_small_12_1.h5")


def mask_detection(roiimage_array):
    roiimage_array = roiimage_array.reshape((1,) + roiimage_array.shape)
    roiimage_array = roiimage_array.astype('float32') / 255
    conf = new_model.predict(roiimage_array)
    return conf


# Load pre-trained model:
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    color_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get dimensions of the input image (to be used later):
    (h, w) = color_img.shape[:2]

    # Create 4-dimensional blob from image:
    blob = cv2.dnn.blobFromImage(color_img, 1.0, (300, 300), [104., 117., 123.], False, False)

    # Set the blob as input and obtain the detections:
    net.setInput(blob)
    detections = net.forward()

    # Initialize the number of detected faces counter detected_faces:
    detected_faces = 0

    # Iterate over all detections:
    for i in range(0, detections.shape[2]):

        # Get the confidence (probability) of the current detection:
        confidence = detections[0, 0, i, 2]
        if confidence < 0.40:
            continue

        # Only consider detections if confidence is greater than a fixed minimum confidence:
        if confidence > 0.6:
            # Increment the number of detected faces:
            detected_faces += 1
            # Get the coordinates of the current detection:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            roi = color_img[(startY): endY, (startX): endX]
            roiimage = cv2.resize(roi, (200, 200))
            # cv2.imwrite("dddd.jpg", roiimage)
            mask_conf = mask_detection(roiimage)
            # print(mask_conf)

            text = ""
            rect_color = ()
            if float(mask_conf) < 0.50:
                # Draw the detection and the confidence:
                text = "{:.3f}%".format(confidence * 100)
                text += " Mask"
                text_color = (255, 0, 0)
                rect_color = (0, 255, 0)
            else:
                text = "{:.3f}%".format(confidence * 100)
                text += " No Mask"
                rect_color = (0, 0, 255)
                text_color = (255, 0, 0)

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), rect_color, 3)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()