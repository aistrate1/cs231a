from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import TensorBoard
import os, sys
import skimage.io as sio
import pandas as pd 
import numpy as np
from scipy import misc
import pickle
from hog import *
def get_X_y_conv_net(X, y, isHog = False):
	labels = []
	images = []
	num_classes = len(y)
	indices = {}
	i = 0
	for i in range(len(y)):
		y_class = y[i]
		if y_class not in indices.keys():		
			indices[y_class] = i
			i += 1

	for i in range(len(X)):
		im = X[i]
		print(im.shape)
		if isHog:
			im = im.reshape(29, 29, 36)
			print(im.shape)
		label = [0 for i in range(10)]
		images.append(im)
		label[indices[y[i]]] = 1
		labels.append(label)
	return np.array(images), np.array(labels)

def get_raw_data(path):
	X = []
	y = []
	dirs = os.listdir(path)
	df = pd.read_csv('train.csv')
	i = 1
	for item in dirs:
		print(str(i) + "/" + str(len(dirs)))
		filename = path + item

		if "resize" in filename and "raw" not in filename:
			im = misc.imread(path + item)
			image_id = item[:16]
			landmark_id = df.loc[df['id'] == image_id, 'landmark_id'].iloc[0]
			X.append(im)
			y.append(landmark_id)
			i += 1
	return np.array(X), np.array(y)

def createModel(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(29, 29, 36)))
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


if __name__ == '__main__':
	hog = True
	file = "1000-10cl"
	if hog:
		with open(file +"-x", 'rb') as fp:
			X = pickle.load(fp)
		with open(file +"-y", 'rb') as fp:
			y = pickle.load(fp)
		X, y = get_X_y_conv_net(X, y, True)
	else:
    		X, y = get_raw_data(file)
		print(X.shape)
		print(y.shape)
		X, y = get_X_y_conv_net(X, y)
        print(y.shape)
    	model = createModel(10)
	tbCallBack = TensorBoard(log_dir="logs", histogram_freq=0, write_graph=True, write_images=True)
	model.fit(X, y, epochs=50, callbacks=[tbCallBack], validation_split = 0.2, verbose=1)
	



