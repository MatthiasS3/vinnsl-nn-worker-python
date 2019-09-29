
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense

red = pd.read_csv("winequality-red.csv", sep=';')
white = pd.read_csv("winequality-white.csv", sep=';')

# Add `type` column to `red` with value 1 and `white` with value 0
red['type'] = 1
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)

x = wines.iloc[:,0:11]
y = np.ravel(wines.type)

# Split the data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state = 42)

# Define the scaler 
scaler = StandardScaler().fit(x_train)
# Scale the train set
x_train = scaler.transform(x_train)
# Scale the test set
x_test = scaler.transform(x_test)

# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))


# Model output shape
model.output_shape
model.summary()
model.get_config()
model.get_weights()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#run Model
model.fit(x_train, y_train,epochs=4, batch_size=8, validation_data=[x_test, y_test], verbose=1)

results = model.evaluate(x_test, y_test)
loss = round(results[0],5)
accuracy = round(results[1],5)
print("Score: ", results)
print("Loss: ", loss)
print("Accuracy: ", accuracy, "\n\n")