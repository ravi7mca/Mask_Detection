# Mask_Detection
This project is helpful in mask detection on faces on Live camera/video streaming. You will get 97% accuracy on validation data. There are three important python files DataPreparation.py, ModelTraining.py and LiveTesting.py.
Mask Dataset are prepared from different internet source.

We have used ResNet for face detection then trained a model on faces to detect Mask on face. Actually we have only 490 images with masks which is not sufficient for CNN so we only focused on faces. First we used image augmentation to increase the images then detected the faces on that and trained a model for mask detection.  

Installation:
Tensorflow, Keras and opencv
