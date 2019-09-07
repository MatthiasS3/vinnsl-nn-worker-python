
import tensorflow as tf
import numpy as np
import os

from keras.datasets import mnist
from keras.layers import *
from keras.activations import *
from keras.models import *
from keras.optimizers import *
from keras.initializers import *
from keras.utils.np_utils import to_categorical
from keras.callbacks import *
from keras import backend as K

from plotting import *

import json
import requests
import datetime


globalEpochsNr = 1
globalId = ""
#vinnslUrl = "http://localhost"
vinnslUrl = "http://vinnsl-service"

class LayerVinnsl:
    def __init__(self, num_nodes, activation_function):
        self.num_nodes = num_nodes
        self.activation_function = activation_function


def on_epoch_end(epoch, _):
    global vinnslUrl
    global globalEpochsNr
    percent = (epoch+1)/globalEpochsNr
    percent = round(percent * 100)
    percent = int(percent)
    global globalId
    #Send Json
    requests.post(vinnslUrl+':8080/vinnsl/create-update/process',
        data=json.dumps(
            {
                'id': globalId,
                'trainingProcess': percent
            }),
        headers={'Content-Type': 'application/json'})

def runMnist(id, data):
    global vinnslUrl
    #Send Json
    requests.put(vinnslUrl+':8080/status/' + id + '/INPROGRESS')
    requests.post(vinnslUrl+':8080/vinnsl/create-update/process',
        data=json.dumps(
            {
                'id': id,
                'trainingProcess': 0
            }),
        headers={'Content-Type': 'application/json'})

    #Set Variables
    lr = float(data["parameters"]["valueparameterOrBoolparameterOrComboparameter"][0]["value"])
    epochs = int(data["parameters"]["valueparameterOrBoolparameterOrComboparameter"][1]["value"])
    global globalEpochsNr
    globalEpochsNr = epochs
    global globalId
    globalId = id


    hiddenLayers = []
    for x in data["structure"]["hidden"]:
        layer = LayerVinnsl(int(x["size"]), "relu")
        hiddenLayers.append(layer)
    
    batch_size = 64

    num_features = int(data["structure"]["input"]["size"])
    num_classes = int(data["structure"]["output"]["size"])

    #Daten setzen
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_size, test_size = x_train.shape[0], x_test.shape[0]

    x_train = x_train.reshape(train_size, num_features)
    x_test = x_test.reshape(test_size, num_features)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    init_w = TruncatedNormal(mean=0.0, stddev=0.05)
    init_b = Constant(value=0.050)

    starttime = datetime.datetime.now()

    #Define the DNN
    model = Sequential()
    #add input layer
    model.add(Dense(hiddenLayers[0].num_nodes, kernel_initializer=init_w,
                    bias_initializer=init_b, input_shape=(num_features,)))
    model.add(Activation("relu"))

    #add hidden layers
    for i in range(1, len(hiddenLayers)):
        #model.add(Dense(x.num_nodes, kernel_initializer=init_w, bias_initializer=init_b, activation=x.activation_function))
        model.add(Dense(
            hiddenLayers[i].num_nodes, kernel_initializer=init_w, bias_initializer=init_b))
        #model.add(Activation(hiddenLayers[i].activation_function))
        model.add(Activation("relu"))

    #add output layer
    model.add(Dense(num_classes, kernel_initializer=init_w,
                    bias_initializer=init_b))
    model.add(Activation("softmax"))

    model.summary()
    
    #Train the DNN
    optimizer = RMSprop(lr=lr)

    sendPercent = LambdaCallback(on_epoch_end=on_epoch_end)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.fit(x=x_train, y=y_train, verbose=1, batch_size=batch_size,
              epochs=epochs, validation_data=[x_test, y_test], callbacks=[sendPercent])
    #Modell speichern
    #model.save("dnn-mnist.h5")

    #Test the DNN
    results = model.evaluate(x_test, y_test)
    loss = round(results[0],5)
    accuracy = round(results[1],5)
    accuracyinPrecent = accuracy * 100
    x = datetime.datetime.now()
    endtime = datetime.datetime.now()
    trainingTime = endtime - starttime
    trainingTime = int(trainingTime.seconds/60)
    print("Score: ", results, "\n\n")

    #Send the final result to vinns-service
    requests.put(vinnslUrl+':8080/status/' + id + '/FINISHED')
    requests.post(vinnslUrl+':8080/vinnsl/create-update/statistic',
        data=json.dumps(
            {
                'id': id,
                'createTimestamp': x.strftime("%d.%m.%Y %I:%M:%S %p"),
                'trainingTime': str(trainingTime),
                'numberOfTraining': 1,
                'lastResult': accuracyinPrecent,
                'bestResult': accuracyinPrecent,
                'epochs': epochs,
                'learningRate': lr,
                'loss': loss,
                'batchSize': 128
            }),
        headers={'Content-Type': 'application/json'})
    
    #Clear Session
    K.clear_session()



