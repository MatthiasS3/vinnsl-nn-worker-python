import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import *

import json
import requests
import datetime


globalEpochsNr = 1
globalId = ""
#vinnslUrl = "http://localhost"
vinnslUrl = "http://vinnsl-service"

def on_epoch_end(epoch, _):
    global globalEpochsNr
    global vinnslUrl
    percent = (epoch+1)/globalEpochsNr
    percent = round(percent * 100)
    percent = int(percent)
    global globalId
    #Send Json
    if (globalEpochsNr % 10) == 0:
        requests.post(vinnslUrl+':8080/vinnsl/create-update/process',
            data=json.dumps(
                {
                    'id': globalId,
                    'trainingProcess': percent
                }),
            headers={'Content-Type': 'application/json'})


class LayerVinnsl:
    def __init__(self, num_nodes, activation_function):
        self.num_nodes = num_nodes
        self.activation_function = activation_function

def runIris(id, data):
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
    

    #Set variables
    lr = float(data["parameters"]["valueparameterOrBoolparameterOrComboparameter"][0]["value"])
    epochs = int(data["parameters"]["valueparameterOrBoolparameterOrComboparameter"][2]["value"])
    global globalEpochsNr
    globalEpochsNr = epochs
    global globalId
    globalId = id


    hiddenLayers = []
    for x in data["structure"]["hidden"]:
        layer = LayerVinnsl(int(x["size"]), data["parameters"]["valueparameterOrBoolparameterOrComboparameter"][1]["value"])
        hiddenLayers.append(layer)
    
    num_features = 4
    num_classes = 3

    #Set data
    iris_data = load_iris() # load the iris dataset

    x = iris_data.data
    y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column

    starttime = datetime.datetime.now()

    # One Hot encode the class labels
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y_)

    # Split the data for training and testing
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

    #############
    #Define the DNN
    model = Sequential()
    #add input layer
    model.add(Dense(hiddenLayers[0].num_nodes, input_shape=(
        num_features,), activation=hiddenLayers[0].activation_function))

    #Define the DNN
    for i in range(1, len(hiddenLayers)):
        #model.add(Dense(x.num_nodes, kernel_initializer=init_w, bias_initializer=init_b, activation=x.activation_function))
        model.add(Dense(hiddenLayers[i].num_nodes, activation=hiddenLayers[i].activation_function))

    #add output layer
    model.add(Dense(num_classes, activation="softmax"))

    model.summary()

    # Adam optimizer with learning rate of 0.001
    optimizer = Adam(lr=lr)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    sendPercent = LambdaCallback(on_epoch_end=on_epoch_end)

    # Train the model
    model.fit(train_x, train_y, verbose=2, batch_size=5,epochs=epochs, callbacks=[sendPercent])

    # Test on unseen data
    results = model.evaluate(test_x, test_y)
    loss = round(results[0],5)
    accuracy = round(results[1],5)
    accuracyinPrecent = accuracy * 100
    x = datetime.datetime.utcnow()
    endtime = datetime.datetime.now()
    trainingTime = endtime - starttime
    print('Final test set loss: {:4f}'.format(results[0]))
    print('Final test set accuracy: {:4f}'.format(results[1]))

    #Send the final result to vinns-service
    requests.put(vinnslUrl+':8080/status/' + id + '/FINISHED')
    requests.post(vinnslUrl+':8080/vinnsl/create-update/statistic',
        data=json.dumps(
            {
                'id': id,
                'createTimestamp': x.strftime("%d.%m.%Y %I:%M:%S %p"),
                'trainingTime': str(trainingTime.seconds),
                'numberOfTraining': 1,
                'lastResult': accuracyinPrecent,
                'bestResult': accuracyinPrecent,
                'epochs': epochs,
                'learningRate': lr,
                'loss': loss
            }),
        headers={'Content-Type': 'application/json'})

    
    #Clear Session
    K.clear_session()
