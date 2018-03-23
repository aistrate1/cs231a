from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from scipy import misc
import numpy as np
import time
from sklearn import svm
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from PIL import Image
import pickle
import skimage.io as sio
from hog import *
from scipy.misc import imrotate
from sklearn.metrics import classification_report,accuracy_score
import urllib
LOGS_DIR = 'logs_dir/'
MODEL_DIR = 'model_dir/'
WEIGHTS_DIR = 'weights_dir/'
BATCH_SIZE = 128
learning_rate = 0.005
max_steps = 1000
img_dim = 256 * 256 * 3
num_classes = 15000
NUM_EPOCHS = 50
id_length = 16
TRAIN_X_LOCATION = "train_x_small"
TRAIN_Y_LOCATION = "train_y_small"
PREPROCESS_DATA = False
NAME = "basic-medium/"
COUNT = 100
IMAGES_PER_CLASS = 10
TOP_10 = [9633, 6051, 6599, 9779, 2061, 5554, 6651, 6696, 5376, 2743]
TOP_5 = [9633, 6051, 6599, 9779, 2061]
counts = {}

#Exploratory analysis - Code publicly available from the Kaggle Competition page at:
#https://www.kaggle.com/codename007/a-very-extensive-landmark-exploratory-analysis'
def exploratory_analysis():
	train_data = pd.read_csv('train.csv')
	submission = pd.read_csv("sample_submission.csv")
	print("Training data size",train_data.shape)
	print(submission.head())

	# Occurance of landmark_id in decreasing order(Top categories)
	print("Most common landmarks")
	temp = pd.DataFrame(train_data.landmark_id.value_counts().head(10))
	temp.reset_index(inplace=True)
	temp.columns = ['landmark_id','count']
	print(temp)

	# Plot the most frequent landmark_ids
	plt.figure(figsize = (9, 8))
	plt.title('Most frequent landmarks')
	sns.set_color_codes("pastel")
	sns.barplot(x="landmark_id", y="count", data=temp,
	            label="Count")
	plt.show()

	#Class distribution
	plt.figure(figsize = (10, 8))
	plt.title('Category Distribuition')
	sns.distplot(train_data['landmark_id'])

	plt.show()

def get_data(num_iter):
	print("Getting batch of data")
	print("--------------------------------")
	df = pd.read_csv('train.csv')
	path = "pictures/train/"
	dirs = os.listdir(path)
	num_files = len(dirs)
	images = []
	labels = []
	for i in range((num_iters - 1) * 10000, num_iters * 10000):
		item = dirs[i]
		filename = path + item
		if "jpg" in filename:
			label = [0 for i in range(num_classes)]
			image = misc.imread(filename)
			images.append(image)
			image_id = item[:16]
			landmark_id = df.loc[df['id'] == image_id, 'landmark_id'].iloc[0]
			print(landmark_id)
			label[landmark_id] = 1
			labels.append(label)
	print("Done gettingg the data")
	print("--------------------------------")
	return np.array(images), np.array(labels)

def prepare_data():
	print("Starting pre-processing the data")
	print("--------------------------------")
	df = pd.read_csv('train.csv')
	path = "pictures/train/"
	dirs = os.listdir(path)
	num_files = len(dirs)
	images = []
	labels = []
	i = 1
	landmark_ids = []
	for item in dirs:
		filename = path + item
		print("Processing x" + str(i) + "/" + str(num_files) + ": " + filename)
		if "resized" in filename:
			image = misc.imread(filename)
			images.append(image)
			image_id = item[:16]
			landmark_id = df.loc[df['id'] == image_id, 'landmark_id'].iloc[0]
			if landmark_id not in landmark_ids:
				landmark_ids.append(landmark_id)
		i += 1
	num_classes = len(landmark_ids)
	print("Total number of classes: " + str(num_classes))
	for item in dirs:
		filename = path + item
		print("Processing y" + str(i) + "/" + str(num_files) + ": " + filename)
		if "resized" in filename:
			label = [0 for i in range(num_classes)]
			image_id = item[:16]
			landmark_id = df.loc[df['id'] == image_id, 'landmark_id'].iloc[0]
			label[landmark_id] = 1
			labels.append(label)
		i += 1

	with open(TRAIN_X_LOCATION, 'wb') as fp:
		pickle.dump(np.array(images), fp)
	with open(TRAIN_Y_LOCATION, 'wb') as fp:
		pickle.dump(np.array(labels), fp)
	print("Done processing data")
	print("--------------------------------")
	return np.array(images), np.array(labels)


def createModel(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(20, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# if PREPROCESS_DATA:
# 	train_x, train_y = prepare_data()
# else:	
# 	with open(TRAIN_X_LOCATION, 'rb') as fp:
# 		train_x = pickle.load(fp)
# 	with open(TRAIN_Y_LOCATION, 'rb') as fp:
# 		train_y = pickle.load(fp)
# print(train_x.shape)
# print(train_y.shape)

def get_top_hog_features(n):
	X = []
	y = []
	df = pd.read_csv('train.csv')
	path = "full_size/"
	for landmark_id in TOP_5:
		print(landmark_id)
		for i in range(n):
			print(str(i))
			image_id = df.loc[df['landmark_id'] == landmark_id, 'id'].iloc[i]
			filename = path + image_id + ".jpg"
			im = sio.imread(filename, True)
			hog_features = compute_hog_features(im, pixels_in_cell, cells_in_block, nbins)
			print(hog_features.shape)
			hog_features = hog_features.flatten()
			X.append(hog_features)
			y.append(landmark_id)
	return X, y

def get_hog_features():
	X = []
	y = []
	path = "10-5cl/"
	dirs = os.listdir(path)
	df = pd.read_csv('train.csv')
	i = 1
	for item in dirs:
		print(str(i) + "/" + str(len(dirs)))
		filename = path + item
		if i == 50000:
			return np.array(X), np.array(y)
		if True:
			im = Image.open(path + item)
			#im = sio.imread(path + item, True)
			im = im.convert('RGB')
			im = im.resize((256,256), Image.ANTIALIAS)
			im.save(path + item + "resize", format='JPEG', quality=90)
			im = sio.imread(path + item + "resize", True)
			image_id = item[:16]
			landmark_id = df.loc[df['id'] == image_id, 'landmark_id'].iloc[0]
			print(landmark_id)
			hog_features = compute_hog_features(im, pixels_in_cell, cells_in_block, nbins)
			print(hog_features.shape)
			#show_hog(im, hog_features, figsize = (18,6))
			hog_features = hog_features.flatten()
			X.append(hog_features)
			y.append(landmark_id)
			i += 1
	return np.array(X), np.array(y)
def get_top_data():
	path = "10-5cl/"
	df = pd.read_csv('train.csv')
	for landmark_id in TOP_10:
		print(landmark_id)
		for i in range(1):
			print(str(i))
			image_id = df.loc[df['landmark_id'] == landmark_id, 'id'].iloc[i]
			url = df.loc[df['landmark_id'] == landmark_id, 'url'].iloc[i]	
			try:
				urllib.urlretrieve(url, path + image_id + ".jpg")
			except Exception:
				pass
def get_raw_pixels():
	X = []
	y = []
	path = "100-10cl/"
	dirs = os.listdir(path)
	df = pd.read_csv('train.csv')
	i = 1
	for item in dirs:
		print(str(i) + "/" + str(len(dirs)))
		filename = path + item

		if "resize" in filename:
			im = Image.open(path + item)
			im = im.convert('RGB')
			im = im.resize((256,256), Image.ANTIALIAS)
			im.save(path + item + "raw", format='JPEG', quality=90)
                        im = sio.imread(path + item + "raw", True)
			image_id = item[:16]
			landmark_id = df.loc[df['id'] == image_id, 'landmark_id'].iloc[0]
			X.append(im.flatten())
			y.append(landmark_id)
			i += 1
	return np.array(X), np.array(y)
if __name__ == '__main__':
    get_top_data()
	# exploratory_analysis()
    pixels_in_cell = 8
    cells_in_block = 2
    nbins = 9
    #X, y = get_raw_pixels()
    get_features = False
    if get_features:
		X, y = get_hog_features()
		with open("10-5cl-x", 'wb') as fp:
			pickle.dump(np.array(X), fp)
		with open("10-5cl-y", 'wb') as fp:
			pickle.dump(np.array(y), fp)
    else:
		with open("1000-5cl-x", 'rb') as fp:
			X = np.array(pickle.load(fp))
		with open("1000-5cl-y", 'rb') as fp:
			y = np.array(pickle.load(fp))
			y = y.reshape(len(y), 1)

    print(X.shape)
    print(y.shape)
    X = np.array(X)
    y = np.array(y)
    y = y.reshape(len(y), 1)
    data_frame = np.hstack((X,y))
    np.random.shuffle(data_frame)
    partition = int(len(y)*0.8)
    X_train, X_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
    y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()


    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy: "+str(accuracy_score(y_test, y_pred)))

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train) 
    y_pred = lin_clf.predict(X_test)
    print("Accuracy: "+str(accuracy_score(y_test, y_pred)))


    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("Accuracy: "+str(accuracy_score(y_test, y_pred)))


    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB().fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print("Accuracy: "+str(accuracy_score(y_test, y_pred)))






