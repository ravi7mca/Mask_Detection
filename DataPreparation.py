"""
Data preparation
Face detection using OpenCV DNN face detector
"""
import cv2 as cv
import numpy as np
import sys
import os
from pathlib import Path
import shutil

base_dir = sys.path[1]
raw_data_path = os.path.join(base_dir, "RawData")
train_data_path = os.path.join(base_dir, "train")

# Data Augmentation only on training data
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
original_image_path = os.path.join(raw_data_path, "raw_training_data\Mask")
augmented_image_path = os.path.join(raw_data_path, "raw_training_data\AugmentedMaskData")

if not os.path.exists(augmented_image_path):
    os.mkdir(augmented_image_path)
else:
    [os.remove(os.path.join(augmented_image_path, f)) for f in os.listdir(augmented_image_path)]

# first copy all original images from original directory to other directory
for index, file in enumerate(os.listdir(original_image_path), start=1):
    shutil.copy(os.path.join(original_image_path, file), os.path.join(augmented_image_path, file))

fname = [os.path.join(augmented_image_path, file) for file in os.listdir(augmented_image_path)]

for index, img_path in enumerate(fname, start=1):
    img = image.load_img(img_path, target_size=(300, 300))
    x = image.img_to_array(img)
    # Reshapes it to (1, 200, 200, 3)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        newImage = image.array_to_img(batch[0])
        fileName = "augmented_Mask_" + str(index) + "_" + str(i) + ".jpg"
        newImage.save(os.path.join(augmented_image_path, fileName))
        # print(x)
        i += 1
        if i % 3 == 0:
            break

# move the augmented image to training directory
raw_training_data_path = os.path.join(raw_data_path, "training_data\Mask")

if not os.path.exists(raw_training_data_path):
    os.mkdir(raw_training_data_path)
else:
    [os.remove(os.path.join(raw_training_data_path, f)) for f in os.listdir(raw_training_data_path)]

for index, file in enumerate(os.listdir(augmented_image_path), start=1):
    shutil.move(os.path.join(augmented_image_path, file), os.path.join(raw_training_data_path, file))

# Load pre-trained model:
net = cv.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")


def data_processing(raw_data_path, processed_data_path, conf):
    for fold in os.listdir(raw_data_path):
        inputPath = os.path.join(raw_data_path, fold)
        savePath = os.path.join(processed_data_path, fold)
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        for file in os.listdir(inputPath):
            # Load image:
            image = cv.imread(os.path.join(inputPath, file))

            # Get dimensions of the input image (to be used later):
            (h, w) = image.shape[:2]

            # Create 4-dimensional blob from image:
            blob = cv.dnn.blobFromImage(image, 1.0, (300, 300), [104., 117., 123.], False, False)

            # Set the blob as input and obtain the detections:
            net.setInput(blob)
            detections = net.forward()

            # Initialize the number of detected faces counter detected_faces:
            detected_faces = 0

            # Iterate over all detections:
            for i in range(0, detections.shape[2]):
                # Get the confidence (probability) of the current detection:
                confidence = detections[0, 0, i, 2]
                # Only consider detections if confidence is greater than a fixed minimum confidence:
                if confidence > conf:
                    # Increment the number of detected faces:
                    detected_faces += 1
                    # Get the coordinates of the current detection:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    roi = image[(startY): endY, (startX): endX]
                    # roiimage = cv2.resize(roi, (200, 200))
                    filename = str(os.path.join(savePath, file)).split('.')[0] + ".jpg"
                    # print(filename)
                    cv.imwrite(filename, roi)


def delete_dirs(root_dir_path):
    for filename in os.listdir(root_dir_path):
        file_path = os.path.join(root_dir_path, filename)
        try:
            if os.path.exists(file_path) and (os.path.isfile(file_path) or os.path.islink(file_path)):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# now lets process the training data
raw_training_data_path = os.path.join(raw_data_path, "training_data")
if os.path.exists(train_data_path):
    delete_dirs(train_data_path)
else:
    os.mkdir(train_data_path)

data_processing(raw_training_data_path, train_data_path, 0.6)

# now lets process the validation data
validation_data_path = os.path.join(base_dir, "validation")
if os.path.exists(validation_data_path):
    delete_dirs(validation_data_path)
else:
    os.mkdir(validation_data_path)

"""for dire in os.listdir(train_data_path):
    label_path = os.path.join(train_data_path, dire)
    savePath = os.path.join(validation_data_path, dire)
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    for index, file in enumerate(os.listdir(label_path), start=1):
        if index < 200:
            shutil.move(os.path.join(label_path,file), os.path.join(savePath, file))"""

raw_validation_data_path = os.path.join(raw_data_path, "validation")
data_processing(raw_validation_data_path, validation_data_path, 0.5)
