import io
import json
from keras.utils.data_utils import get_file

with open('tweets.json') as f:
    tweets = json.load(f)

text = ""
f = open("tweets.txt", "w+")

for tweet in tweets:
    text += tweet["text"] + " "

print(len(text))
f.write(text)
f.close()

#path = get_file('tweets.txt', origin='')

with io.open('tweets.txt', encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))
