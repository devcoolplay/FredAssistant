
# imports
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import sys
import numpy
import tflearn
import os
import tensorflow
import random
import json
import pickle

# init variable to determine wether the bot should be started after training
startBotWhenFinished = "False"

# check wether the arguments are correct
if len(sys.argv) == 2:
    startBotWhenFinished = sys.argv[1]
elif len(sys.argv) == 1:
    pass
elif startBotWhenFinished != "-s":
    print("Invalid arguments!\nValid arguments are:\n   -s     to start bot after training")
    exit(1)
else:
    print("Invalid arguments!\nValid arguments are:\n   -s     to start bot after training")
    exit(1)

if startBotWhenFinished != "-s":
    print("Invalid arguments!\nValid arguments are:\n   -s     to start bot after training")
    exit(1)

# open the intents.json file
with open("intents.json") as file:
    data = json.load(file)

# init vars for training data
words = []
labels = []
docs_x = []
docs_y = []

# prepare data for training
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # tokenize words and load data from file
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# stem words
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
# sort words
words = sorted(list(set(words)))

# sort labels
labels = sorted(labels)

# define input and output data
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

# prepare input and output
for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

# save training data
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

# set up neural network
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# save training data
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

# start the bot if argument "-s" is applied
if startBotWhenFinished == "-s":
    os.system('cmd /c "py bot.py"')