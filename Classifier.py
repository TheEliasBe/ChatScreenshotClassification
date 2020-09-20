import numpy as np
import os
from skimage import data, io, color, transform
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# load images
all_chat_file_names = os.listdir("media/convo")
other_file_names = os.listdir("media/not_convo")

# assign labels
chat_files = np.transpose(np.array([all_chat_file_names]))
chat_labels = np.transpose(np.array([[1.0]*len(chat_files)]))
chat_files = np.concatenate((chat_files, chat_labels), axis=1)

non_chat_files = np.transpose(np.array([other_file_names]))
non_chat_labels = np.transpose(np.array([[0.0]*len(non_chat_files)]))
non_chat_files = np.concatenate((non_chat_files, non_chat_labels), axis=1)

# combine files
files = np.concatenate((chat_files, non_chat_files), axis=0)

# generate the resized and grey scaled images
def resize_image():
    # find average size
    sum_height = 0
    sum_width = 0
    for img_file in files:
        try:
            image = io.imread("media/convo/"+img_file[0])
        except Exception as exc:
            pass
        try:
            image = io.imread("media/not_convo/"+img_file[0])
        except Exception as exc:
            pass

        sum_height += np.shape(image)[0]
        sum_width += np.shape(image)[1]

    avg_height = sum_height//len(files)
    avg_width = sum_width//len(files)
    print("Average height", avg_height)
    print("Average width", avg_width)

    # rescale every image to average size
    i = 0
    for img_file in files:
        try:
            image = io.imread("media/convo/"+img_file[0])
        except Exception as exc:
            pass
        try:
            image = io.imread("media/not_convo/"+img_file[0])
        except Exception as exc:
            pass

        image = color.rgb2gray(image)
        image_rescaled = transform.resize(image, (avg_height, avg_width))
        io.imsave(fname="media/resized/"+img_file[0], arr=image_rescaled)
        i += 1
        if i == 10:
            pass

# read resized and greyed images an np-arrays
pixels = []
for img_file in files:
    img = io.imread("media/resized/"+img_file[0])
    img = transform.rescale(img, 0.1)
    img = img.flatten()
    pixels.append(img)

pixels = np.array(pixels, dtype=np.float)

files = np.concatenate((files, pixels), axis=1)

# remove file name, not necessary
files = files[:,1:]

# set dtype to float64
files = files.astype('float64')

# split data into test and train
X_train, X_test, y_train, y_test = train_test_split(files[:,2:16379], files[:,0], test_size=0.2, random_state=42)

print(X_train)
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)
print(y_pred)
print(accuracy_score(y_test, y_pred))

