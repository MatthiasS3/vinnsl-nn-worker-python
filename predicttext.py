from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import *
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def predictText(text, textLength, diversity):
    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    # Function invoked at end of each epoch. Prints generated text.
    model = load_model("tweets6.h5")
    maxlen = 40
    print()
    print('----- Generating text  -----')

    start_index = random.randint(0, len(text) - maxlen - 1)
    sentence = text[start_index: start_index + maxlen]
    #for diversity in [0.2, 0.5, 1.0, 1.2]:
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    #print('----- Generating with seed: "' + sentence + '"')

    for i in range(textLength):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
    
    return generated



with io.open('tweets.txt', encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

textLength = 400
diversity = 1.0

output = predictText(text, textLength, diversity)
print("--")
print("--")
print(output)
print("--")
print("--")
