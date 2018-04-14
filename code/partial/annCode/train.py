#!/usr/bin/env python3
import pandas as pd
import tensorflow as tf
import numpy as np


def makeOneHot(yVal):
    yOneHot = []
    for y in yVal:
        if y == 'Iris-virginica':
            yOneHot.append([1, 0, 0])
        elif y == 'Iris-versicolor':
            yOneHot.append([0, 1, 0])
        elif y == 'Iris-setosa':
            yOneHot.append([0, 0, 1])
    return yOneHot


def getData():
    dataPath = "./data/iris.csv"

    data = pd.read_csv(dataPath, header=None)
    data.columns = ["SepalLength", "SepalWidth",
                    "PetalLength", "PetalWidth",
                    "Class"]

    data = data.sample(frac=1)

    xDat = data.iloc[:, 0:4]
    yDat = data.iloc[:, 4]

    xTrain = xDat.iloc[:120]
    xTest = xDat.iloc[120:]

    #  Min max normalization
    minVec = xTrain.min()
    maxVec = xTrain.max()
    meanVec = xTrain.mean()

    xTrain = ((xTrain - meanVec)/(maxVec - minVec)).values
    xTest = ((xTest - meanVec)/(maxVec - minVec)).values

    #  Convert to one-hot vectors
    yTrain = makeOneHot(yDat.iloc[:120])
    yTest = makeOneHot(yDat.iloc[120:])

    return yTrain, yTest, xTrain, xTest


class nnModel(object):

    def __init__(self, learningRate):
        XSHAPE = 4
        YSHAPE = 3

        #  Define inputs
        self.xDat = tf.placeholder("float", shape=[None, XSHAPE])
        self.yDat = tf.placeholder("float", shape=[None, YSHAPE])

        #  Define Weights
        H1 = 8
        H2 = 5

        self.w1 = self.init_weights((XSHAPE, H1))
        self.w2 = self.init_weights((H1, H2))
        self.w3 = self.init_weights((H2, YSHAPE))


        #  Define equations relating weights(Forward Prop)
        # ========================
        """ Define Two fully connected layers here """

        #  h1 =
        #  h2 =
        raise NotImplementedError  # Remove when done
        # ========================

        h3 = tf.matmul(h2, self.w3)
        self.yPred = tf.nn.softmax(h3)
        self.predictClass = tf.argmax(self.yPred, axis=1)

        #  Define backpropagation and Loss
        # ========================
        """  Define expression for Loss here """
        #  self.loss =
        raise NotImplementedError  # Remove when done
        # ========================

        self.gradUpdater =  \
            tf.train.GradientDescentOptimizer(learningRate).minimize(self.loss)

    def init_weights(self, shape):
        """ Weight initialization """
        weights = tf.random_normal(shape, stddev=0.1)
        return tf.Variable(weights)


def main():
    LEARNING_RATE = 0.1
    EPOCHS = 300

    #  1) Read data
    #  2) Normalize data features
    #  3) Convert labels to one-hot representation
    yTrain, yTest, xTrain, xTest = getData()

    #  4) Define Neural network architecture(forward)
    #  5) Define loss function and update function(backward)
    model = nnModel(learningRate=LEARNING_RATE)

    #  Initialize session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    #  6) Split data into batches and run epochs
    #  7) Call the grad update for every epoch
    for epoch in range(EPOCHS):
        #  Stochastic descent, feed one point at a time
        for i in range(len(xTrain)):

            #  Apply one step of gradient descent
            sess.run(
                model.gradUpdater,
                feed_dict={model.xDat: xTrain[i: i + 1], model.yDat: yTrain[i: i + 1]})

        modeltrainPred, loss1 = sess.run(
            [model.predictClass, model.loss],
            feed_dict={model.xDat: xTrain, model.yDat: yTrain})
        modeltestPred, loss2 = sess.run(
            [model.predictClass, model.loss],
            feed_dict={model.xDat: xTest, model.yDat: yTest})

        train_accuracy = np.mean(np.argmax(yTrain, axis=1) == modeltrainPred)
        test_accuracy = np.mean(np.argmax(yTest, axis=1) == modeltestPred)
        train_loss = np.mean(loss1)
        test_loss = np.mean(loss2)

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

        print("Epoch = %d, train loss = %.2f, test loss = %.2f\n"
              % (epoch + 1, train_loss, test_loss))

    sess.close()


if __name__ == '__main__':
    main()


