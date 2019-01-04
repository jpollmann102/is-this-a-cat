import matplotlib.pyplot as plt
import numpy as np
import os.path
import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import preprocess as pre

LR = 1e-3
imgSize = 50

# if os.path.exists('data/test_data.npy'):
#     testData = np.load('data/test_data.npy')
# else:
#     testData = pre.createTestData()

predictData = pre.createPredictData('data/predict')

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

model.load('models/is-this-a-cat.tflearn')

fig = plt.figure(figsize=(16, 12))

for num, data in enumerate(predictData[:9]):

    imgNum = data[1]
    imgData = data[0]

    y = fig.add_subplot(4, 4, num + 1)
    orig = imgData
    data = imgData.reshape(imgSize, imgSize, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label='Not a Cat'
    else:
        str_label='Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()
