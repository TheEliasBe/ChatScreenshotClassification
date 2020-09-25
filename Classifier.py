import numpy as np
import os

from imblearn.over_sampling import RandomOverSampler
from skimage import data, io, color, transform
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn
import imblearn


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

print("Number of samples", len(chat_files)+len(non_chat_files), sep="\n")
print("Ratio Class 'Chat' of all samples", len(chat_files)/(len(chat_files)+len(non_chat_files)), sep="\n")

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
        image_rescaled = transform.rescale(image_rescaled, 0.1)
        io.imsave(fname="media/resized/"+img_file[0], arr=image_rescaled)
        i += 1
        if i == 10:
            pass


# read resized and greyed images an np-arrays
pixels = []
for img_file in files:
    img = io.imread("media/resized/"+img_file[0])
    img = img.flatten()
    pixels.append(img)


pixels = np.array(pixels, dtype=np.float)
files = np.concatenate((files, pixels), axis=1)

# convert to data frame
df = pd.DataFrame(files)

# assign dtypes
df[0] = df[0].astype('object')
df.iloc[:,1:] = df.iloc[:,1:].astype('float')

# set X and y
X = df.iloc[:,2:].astype('float')
y = df[1].astype('int')
print(X)

# random over sample the non-chat files
oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)

# split data into test and train
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)
y_train = np.asarray(y_train,dtype=np.float64)
y_test = np.asarray(y_test,dtype=np.float64)



# create classifier
clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, y_pred), sep="\n")
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, range(2), range(2))
sn.heatmap(df_cm, annot=True)

# find & display all misclassified instances
a = np.hstack((np.c_[y_pred], np.c_[y_test], X_test))
print(a[:,0:2])
for line in a:
    if line[0] == 0 and line[1] == 1:
        # classified as non-chat but actually chat
        i = np.reshape(line[2:], (159,103))
        i = i.astype('float')
        io.imshow(arr=i)
        io.show()

    if line[0] == 1 and line[1] == 0:
        # classified as chat but actually non-chat
        i = np.reshape(line[2:], (159,103))
        i = i.astype('float')
        io.imshow(arr=i)
        io.show()