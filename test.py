import os
import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread

import pickle

import os
import imageio
import matplotlib.pyplot as plt

new_image_size = (224, 224, 3)

# set the directory containing the images
images_dir = './test_dataset/training'

current_id = 0

# for storing the foldername: label,
label_ids = {}

# for storing the images data and labels
images = []
labels = []

for root, _, files in os.walk(images_dir):
    for file in files:
        if file.endswith(('png', 'jpg', 'jpeg')):
            # path of the image
            path = os.path.join(root, file)

            # get the label name
            label = os.path.basename(root).replace(
                " ", ".").lower()

            # add the label (key) and its number (value)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            # save the target value
            labels.append(current_id-1)

            # load the image, resize and flatten it
            image = imread(path)
            image = resize(image, new_image_size)
            images.append(image.flatten())


print(label_ids)

# save the labels for each fruit as a list
categories = list(label_ids.keys())

pickle.dump(categories, open("faces_labels.pk", "wb"))

df = pd.DataFrame(np.array(images))
df['Target'] = np.array(labels)

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x_train, x_test, y_train, y_test = \
    train_test_split(x, y,
        test_size = 0.30,    # 10% for test
        random_state=77,
        stratify = y)

# trying out the various parameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma' : [0.0001, 0.001, 0.1,1],
    'kernel' : ['rbf', 'poly']
}

svc = svm.SVC(probability=True)
print("Starting training, please wait ...")

# exhaustive search over specified hyper parameter
# values for an estimator
model = GridSearchCV(svc,param_grid)
model.fit(x_train, y_train)

# print the parameters for the best performing model
print(model.best_params_)

y_pred = model.predict(x_test)
print(f"The model is {accuracy_score(y_pred,y_test) * 100}% accurate")

pickle.dump(model, open('faces_model.pickle','wb'))
