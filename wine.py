import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

def runWine(id, data):
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
    epochs = int(data["parameters"]["valueparameterOrBoolparameterOrComboparameter"][1]["value"])
    global globalEpochsNr
    globalEpochsNr = epochs
    global globalId
    globalId = id


    hiddenLayers = []
    for x in data["structure"]["hidden"]:
        layer = LayerVinnsl(int(x["size"]), data["parameters"]["valueparameterOrBoolparameterOrComboparameter"][1]["value"])
        hiddenLayers.append(layer)
    
    num_features = int(data["structure"]["input"]["size"])
    num_classes = int(data["structure"]["output"]["size"])
    print(data)

    #Set data
    red = pd.read_csv("winequality-red.csv", sep=';')
    white = pd.read_csv("winequality-white.csv", sep=';')

    # Add `type` column to `red` with value 1 and `white` with value 0
    red['type'] = 1
    white['type'] = 0

    # Append `white` to `red`
    wines = red.append(white, ignore_index=True)

    x = wines.iloc[:,0:11]
    y = np.ravel(wines.type)


    starttime = datetime.datetime.now()

    # Split the data for training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state = 42)

    # Define the scaler 
    scaler = StandardScaler().fit(x_train)
    # Scale the train set
    x_train = scaler.transform(x_train)
    # Scale the test set
    x_test = scaler.transform(x_test)


    #############
    #Define the DNN
    model = Sequential()
    #add input layer
    #model.add(Dense(hiddenLayers[0].num_nodes, input_shape=(num_features,), activation=hiddenLayers[0].activation_function))
    model.add(Dense(12, activation='relu', input_shape=(11,)))

    #Define the DNN
    for i in range(1, len(hiddenLayers)):
        #model.add(Dense(x.num_nodes, kernel_initializer=init_w, bias_initializer=init_b, activation=x.activation_function))
        model.add(Dense(hiddenLayers[i].num_nodes, activation=hiddenLayers[i].activation_function))
        print("Dense-Layer:", hiddenLayers[i].num_nodes)

    #add output layer
    #model.add(Dense(num_classes, activation="sigmoid"))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    sendPercent = LambdaCallback(on_epoch_end=on_epoch_end)

    # Train the model
    model.fit(x_train, y_train,epochs=epochs, batch_size=8, validation_data=[x_test, y_test], verbose=1, callbacks=[sendPercent])

    # Test on unseen data
    results = model.evaluate(x_test, y_test)
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
                'loss': loss
            }),
        headers={'Content-Type': 'application/json'})

    
    #Clear Session
    K.clear_session()
