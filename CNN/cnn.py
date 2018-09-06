import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt

TRAIN_DIR = './asl-alphabet/train'
TEST_DIR = './asl-alphabet/test'
IMG_WIDTH = 200
IMG_HEIGHT = 200
LR = 1e-6
MODEL_NAME = 'asl-alphabet(A,B,C,D,epochs30,LR1e-6)-{}-{}.model'.format(LR, '2conv-basic')

def switch_label(x):
	return {
		'A': [1,0,0,0],
		'B': [0,1,0,0],
		'C': [0,0,1,0],
		'D': [0,0,0,1],
	}[x]

def switch_number(x):
	return {
		0: 'A',
		1: 'B',
		2: 'C',
		3: 'D'
	}[x]

def label_img(img):
	label = img.split('.')[0][0]
	return switch_label(label)

def process_data(img,label):
	img = cv2.resize(img, (IMG_HEIGHT,IMG_WIDTH))
	npImg = np.array(img)
	npLabel = np.array(label)
	return npImg, npLabel

def create_train_data():
	if 'train_data.npy' in os.listdir():
		training_data = np.load('train_data.npy')
	else:
		training_data = []
		for img in tqdm(os.listdir(TRAIN_DIR)):
			label = label_img(img)
			path = os.path.join(TRAIN_DIR,img)
			img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
			npImg, npLabel = process_data(img, label)
			training_data.append([npImg, npLabel])
		shuffle(training_data)
		np.save('train_data.npy', training_data)
	return training_data

def process_test_data():
	if 'test_data.npy' in os.listdir():
		testing_data = np.load('test_data.npy')
	else:
		testing_data = []
		for img in tqdm(os.listdir(TEST_DIR)):
			path = os.path.join(TEST_DIR,img)
			img_ID = img.split('.')[0]
			img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
			npImg, npLabel = process_data(img, img_ID)
			testing_data.append([npImg, img_ID])
		shuffle(testing_data)
		np.save('test_data.npy', testing_data)
	return testing_data

def create_model(train_data=None):
	
	tf.reset_default_graph()

	convnet = input_data(shape=[None, IMG_HEIGHT, IMG_WIDTH, 1], name='input')

	convnet = conv_2d(convnet, 32, 3, padding='valid', activation='relu')
	convnet = max_pool_2d(convnet, 2, padding='valid', strides=2)

	convnet = conv_2d(convnet, 64, 3, padding='valid', activation='relu')
	convnet = max_pool_2d(convnet, 2, padding='valid', strides=2)

	convnet = conv_2d(convnet, 128, 3, padding='valid', activation='relu')
	convnet = max_pool_2d(convnet, 2, padding='valid', strides=2)

	convnet = conv_2d(convnet, 256, 5, padding='valid', activation='relu')
	convnet = max_pool_2d(convnet, 2)

	convnet = conv_2d(convnet, 256, 5, activation='relu')
	convnet = max_pool_2d(convnet, 3)

	convnet = conv_2d(convnet, 256, 3, activation='relu')
	convnet = max_pool_2d(convnet, 3)

	convnet = conv_2d(convnet, 512, 3, activation='relu')
	convnet = max_pool_2d(convnet, 2)

	convnet = fully_connected(convnet, 1024, activation='relu')
	convnet = dropout(convnet, 0.8)

	convnet = fully_connected(convnet, 4, activation='softmax')
	convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(convnet, tensorboard_dir='log')

	if os.path.exists('{}.meta'.format(MODEL_NAME)):
		model.load(MODEL_NAME)
		print('model loaded!')
		return model

	else:

		if train_data == None:
			train_data = create_train_data()

		test_size = int(0.2*len(train_data))

		train = train_data[:-test_size]
		test = train_data[-test_size:]


		X = np.array([i[0] for i in train]).reshape(-1,IMG_HEIGHT, IMG_WIDTH, 1)
		y = [i[1] for i in train]

		X_test = np.array([i[0] for i in test]).reshape(-1,IMG_HEIGHT, IMG_WIDTH, 1)
		y_test = [i[1] for i in test]

		model.fit({'input': X}, {'targets': y}, n_epoch=30, validation_set=({'input': X_test}, {'targets': y_test}), snapshot_step=100, show_metric=True, run_id=MODEL_NAME)

		model.save(MODEL_NAME)
		print('model created!')
		return model

def run_test_data(test_data, model):

	fig=plt.figure()

	for num,data in enumerate(test_data[30:42]):
		img_num = data[1]
		img_data = data[0]

		y = fig.add_subplot(3,4,num+1)
		orig = img_data
		data = img_data.reshape(IMG_HEIGHT,IMG_WIDTH,1)
		model_out = model.predict([data])[0]

		str_label = switch_number(np.argmax(model_out))
		    
		y.imshow(orig,cmap='gray')
		plt.title(str_label)
		y.axes.get_xaxis().set_visible(False)
		y.axes.get_yaxis().set_visible(False)
	plt.show()

def predict_data(img_data, model):
	data = img_data[0].reshape(IMG_HEIGHT,IMG_WIDTH,1)
	model_out = model.predict([data])[0]
	str_label = switch_number(np.argmax(model_out))
	return str_label