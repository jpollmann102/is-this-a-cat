import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

trainDir = 'data/train'
testDir = 'data/test'
imgSize = 50

def createLabel(imageName):
    # create outputs based on image names
    wordLabel = imageName.split('.')[-3]
    if wordLabel == 'cat':
        return np.array([1,0])
    elif wordLabel == 'dog':
        return np.array([0,1])

def createTrainData():
    trainingData = []
    for img in tqdm(os.listdir(trainDir)):
        path = os.path.join(trainDir, img)
        imgData = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        imgData = cv2.resize(imgData, (imgSize, imgSize))
        trainingData.append([np.array(imgData),
                            createLabel(img)])

    shuffle(trainingData)
    np.save('data/train_data.npy', trainingData)
    return trainingData

def createTestData():
    testData = []
    for img in tqdm(os.listdir(testDir)):
        path = os.path.join(testDir, img)
        imgNum = img.split('.')[0]
        imgData = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        imgData = cv2.resize(imgData, (imgSize, imgSize))
        testData.append([np.array(imgData), imgNum])

    shuffle(testData)
    np.save('data/test_data.npy', testData)
    return testData

def createPredictData(dataPath):
    predictData = []
    for img in tqdm(os.listdir(dataPath)):
        path = os.path.join(dataPath, img)
        imgNum = img.split('.')[0]
        imgData = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        imgData = cv2.resize(imgData, (imgSize, imgSize))
        predictData.append([np.array(imgData), imgNum])

    shuffle(predictData)
    return predictData
