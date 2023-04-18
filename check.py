import cv2
import os
import pickle
import numpy as np
import pickle

from PIL import Image
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array
from keras_vggface import utils

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# returns a compiled model identical to the previous one
model = load_model('vgg16_face_cnn_model.h5')

# dimension of images
image_width = 224
image_height = 224

# load the training labels
face_label_filename = 'vgg16-face-labels.pickle'
with open(face_label_filename, "rb") as \
        f:
    class_dictionary = pickle.load(f)

class_list = [value for _, value in class_dictionary.items()]

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

path = os.getcwd() + '//facetest'

for image in os.listdir(path):
    frame = cv2.imread(path + "//" + image)
    try:
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    except:
        continue
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        secondStartY = int(abs(endY-startY)/2)
        secondColor = (10, 30, 159)
        print(pred)
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        # Save the extracted image as a separate file
        rect_img = frame[startY:endY-secondStartY, startX:endX]
        size = (image_width, image_height)
        resized_image = cv2.resize(rect_img, size)
        x = img_to_array(resized_image)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)

        predicted_prob = model.predict(x)
        print(predicted_prob)
        # predicted_classes = np.argmax(predicted_prob, axis=1)
        # print(predicted_classes)
        print(predicted_prob[0].argmax())
        print("Predicted face: " + class_list[predicted_prob[0].argmax()])
        print("============================\n")
        break


# preds = model.predict(x)
# predicted_classes = np.argmax(preds, axis=1)
# class_labels = list(train_generator.class_indices.keys())
# predicted_labels = [class_labels[i] for i in predicted_classes]
# print('Predicted:', utils.decode_predictions(
#     np.expand_dims(preds, axis=0), top=5))


# for i in range(1, 13):
#     test_image_filename = f'./facetest/{i}.jpg'
#     # load the image
#     imgtest = cv2.imread(test_image_filename, cv2.IMREAD_COLOR)
#     image_array = np.array(imgtest, "uint8")

#     # get the faces detected in the image
#     faces = facecascade.detectMultiScale(imgtest,
#                                          scaleFactor=1.1, minNeighbors=5)

#     # if not exactly 1 face is detected, skip this photo
#     if len(faces) != 1:
#         print(f'---We need exactly 1 face;')
#         print(f'photo skipped---')
#         continue

#     for (x_, y_, w, h) in faces:
#         # draw the face detected
#         face_detect = cv2.rectangle(
#             imgtest, (x_, y_), (x_+w, y_+h), (255, 0, 255), 2)
#         plt.imshow(face_detect)
#         plt.show()

#         # resize the detected face to 224x224
#         size = (image_width, image_height)
#         roi = image_array[y_: y_ + h, x_: x_ + w]
#         resized_image = cv2.resize(roi, size)

#         # prepare the image for prediction
#         x = img_to_array(resized_image)
#         x = np.expand_dims(x, axis=0)
#         x = utils.preprocess_input(x, version=1)

#         # making prediction
#         predicted_prob = model.predict(x)
#         print(predicted_prob)
#         print(predicted_prob[0].argmax())
#         print("Predicted face: " + class_list[predicted_prob[0].argmax()])
#         print("============================\n")
