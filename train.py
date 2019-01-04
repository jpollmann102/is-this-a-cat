import numpy as np
import os.path
import tensorflow as tf

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import preprocess as pre

LR = 1e-3
imgSize = 50

saver = tf.train.Saver()

if os.path.exists('data/train_data.npy'):
    trainData = np.load('data/train_data.npy')
else:
    trainData = pre.createTrainData()

if os.path.exists('data/test_data.npy'):
    testData = np.load('data/test_data.npy')
else:
    testData = pre.createTestData()

train = trainData[:-500]
test = trainData[-500:]

xTrain = np.array([i[0] for i in train]).reshape(-1, imgSize, imgSize, 1)
yTrain = [i[1] for i in train]

xTest = np.array([i[0] for i in test]).reshape(-1, imgSize, imgSize, 1)
yTest = [i[1] for i in test]

tf.reset_default_graph()

# resized images into imgSize x imgSize x 1 matrices
convnet = input_data(shape=[None, imgSize, imgSize, 1], name='input')

# add convolutional layer with 32 filters, stride = 5
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# add convolutional layer with 64 filters, stride = 5
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# add convolutional layer with 128 filters, stride = 5
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# add convolutional layer with 64 filters, stride = 5
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# add convolutional layer with 32 filters, stride = 5
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# add fully connected layer with 1024 neurons
convnet = fully_connected(convnet, 1024, activation='relu')

# add dropout layer with probability of 0.8
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                    loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

model.save('models/is-this-a-cat.tflearn')

model.fit({'input': xTrain}, {'targets': yTrain}, n_epoch=10,
          validation_set=({'input': xTest}, {'targets': yTest}),
          snapshot_step=500, show_metric=True, run_id='is-this-a-cat-convnet')

model.save('models/is-this-a-cat.tflearn')
