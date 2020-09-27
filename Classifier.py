import numpy as np
import os

from imblearn.over_sampling import RandomOverSampler
from skimage import data, io, color, transform
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import pickle

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

#print("Number of samples", len(chat_files)+len(non_chat_files), sep="\n")
#print("Ratio Class 'Chat' of all samples", len(chat_files)/(len(chat_files)+len(non_chat_files)), sep="\n")

# width, height constant as an average of ca. 400 images
WIDTH = 1024
HEIGHT = 1607

# combine files
files = np.concatenate((chat_files, non_chat_files), axis=0)

# generate the resized and grey scaled images
def resize_image(avg_width, avg_height):
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
        image = transform.resize(image, (avg_height, avg_width))
        image = transform.rescale(image, 0.1)
        io.imsave(fname="media/resized/"+img_file[0], arr=image)
        i += 1
        if i == 10:
            pass

def get_avg_size():
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
    return avg_width, avg_height

def save_hog(width, height):
    for img_file in files:
        print("Computing HOG for", img_file[0])
        try:
            image = io.imread("media/convo/"+img_file[0])
        except Exception as exc:
            pass
        try:
            image = io.imread("media/not_convo/"+img_file[0])
        except Exception as exc:
            pass

        image = color.rgb2gray(image)
        print(image.shape)
        image = transform.resize(image, (height, width))
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),cells_per_block = (1, 1), visualize = True, multichannel = False)
        np.save("media/hog/"+img_file[0], hog_image)

def train_model(use_hog=False):
    global files
    # read resized and greyed images an np-arrays
    pixels = []
    if use_hog:
        files = os.listdir("media/hog")
        for img_file in files:
            print("Reading HOG", img_file)
            img = np.load("media/hog/"+img_file)
            img = img.flatten()
            pixels.append(img)
    else:
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

    # random over sample the non-chat files
    oversample = RandomOverSampler(sampling_strategy=0.5)
    X_over, y_over = oversample.fit_resample(X, y)
    X_over = X
    y_over = y

    X_over = X_over.astype('int')

    # scale data from [0,255] -> [0,1]
    scaler = StandardScaler()
    X_over = scaler.fit_transform(X=X_over)

    # split data into test and train
    X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)
    y_train = np.asarray(y_train,dtype=np.float64)
    y_test = np.asarray(y_test,dtype=np.float64)

    # create classifier
    clf = MLPClassifier(hidden_layer_sizes=(100, 100, 10), max_iter=1000, learning_rate='adaptive')
    with open("pickle_model_90_1.pkl", 'rb') as file:
        clf2 = pickle.load(file)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy", accuracy_score(y_test, y_pred), sep="\n")
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    pkl_filename = "pickle_model_90_1.pkl"
    with open(pkl_filename, 'wb') as file:
       pickle.dump(clf, file)

    # find & display all misclassified instances
    a = np.hstack((np.c_[y_pred], np.c_[y_test], X_test))


    # for line in a:
    #     if line[0] == 0 and line[1] == 1:
    #         # classified as non-chat but actually chat
    #         i = np.reshape(line[2:], (159,103))
    #         i = i.astype('float')
    #         io.imshow(arr=i)
    #         io.show()
    #
    #     if line[0] == 1 and line[1] == 0:
    #         # classified as chat but actually non-chat
    #         i = np.reshape(line[2:], (159,103))
    #         i = i.astype('float')
    #         io.imshow(arr=i)
    #         io.show()

def classify(img_arr, model_file_name):
    for index in range(len(img_arr)):
        img_arr[index] = color.rgb2gray(img_arr[index])
        img_arr[index] = transform.resize(img_arr[index], (HEIGHT, WIDTH))
        img_arr[index] = transform.rescale(img_arr[index], 0.1)

        io.imshow(img_arr[index])
        io.show()

        scaler = StandardScaler()
        img_arr[index] = scaler.fit_transform(X=img_arr[index])
        img_arr[index] = img_arr[index].flatten()

        print(img_arr[index].dtype)

    # Load from file
    with open(model_file_name, 'rb') as file:
        pickle_model = pickle.load(file)

    # Calculate the accuracy score and predict target values
    y_pred = pickle_model.predict(img_arr)
    y_pred_proba = pickle_model.predict_proba(img_arr)
    return (y_pred, y_pred_proba)

i1 = io.imread("media/convo/o2cvz4sbmrn51.jpg")
i2 = io.imread("media/not_convo/0qkepwvr0q641.jpg")
i3 = io.imread("media/not_convo/1bar6xw825p51.png")
i4 = io.imread("media/resized/2s7q2tbdd1o51.png")
i5 = io.imread("media/downscaled.jpg")
i6 = io.imread("media/hg.png")

train_model()
# class_id, prob = classify([i6], "pickle_model_90_1.pkl")
#
# if class_id[0] == 0:
#     print("Chat Screenshot. Confidence: ", f'{prob[0][0]*100:.2f}', '%')
# else:
#     print("No Chat Screenshot. Confidence: ", f'{prob[0][1]*100:.2f}', '%')


