# This is the Fred assistant bot in a very early stage.
# Updates with customization process coming soon.

# imports
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import os
import tflearn
import keyboard
import tensorflow
import random
import speech_recognition as sr
import json
import pickle

# initialize the speech recognizer engine
speech_engine = sr.Recognizer()
# set a variable to check wether the bot is already trained
isTrained = True
# init variable to check wether the training data got changed by the bot
dataChanged = False
# init variable to check if the user wants the bot to train with its collected data 
train = False

# init variables for hotkey detection
current = set()
pressed = False

# set the pressed variable to true
def setPressed():
    global pressed
    pressed = True

# add a hotkey to listen to
keyboard.add_hotkey('alt + s', setPressed)

# open the intents.json file in read mode to read the data
with open("intents.json", "r") as file:
    data = json.load(file)

# try to open training data. if training data exists go on, and if not close the program with an error message
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    isTrained = False

# initialize nural network
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

if isTrained: 
    # load the training data
    model.load("model.tflearn")
else:
    print("No trained model found!")

# compare input words with words stored in data
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

# start to understand what the user wants to happen, when hotkey is pressed 
def main():
    # tell the user that the bot is ready
    print("Start talking with Fred! (type \"quit\" to stop)")
    # init vars
    global pressed
    previousTag = "nothing"
    lastSaid = "nothing"
    # start endless loop
    while True:
        # check wether the hotkey is pressed, and proceed when it is
        if pressed:
            # define an input variable to store the user's input
            inp = "nothing"
            # try to listen to the user and understand the user
            try:
                # start the mic and listen to what the user is saying
                with sr.Microphone() as micro:
                    print("Recording...")
                    audio = speech_engine.record(micro, duration=3)
                    print("Recognition...")
                    text = speech_engine.recognize_google(audio, language="de-DE")
                    print(text)

                    inp = text
                    # reset pressed, so it won't start this process for no reason
                    pressed = False

                    # interpret what the user wants to happen
                    results = model.predict([bag_of_words(inp, words)])[0]
                    results_index = numpy.argmax(results)
                    tag = labels[results_index]
                    
                    # init a variable to tell the program wether a new expression should be saved in the intents.json file
                    save = True

                    # check wether the ai is confident enough with its interpretations
                    if results[results_index] > 0.7:
                        # execute the things that the ai understood
                        if tag == "openbrowser":
                            os.system('cmd /c "start brave.exe"')
                            print("Brave browser has been started")
                        elif tag == "waswrong":
                            save = False
                        elif tag == "openspotify":
                            os.system('cmd /c "start spotify.exe"')
                            print("Spotify has been opened")
                        elif tag == "openfiles":
                            os.system('cmd /c "start explorer.exe"')
                            print("explorer has been opened")
                        elif tag == "openvscode":
                            os.system('cmd /c "code"')
                            print("visual studio code has been opened")
                        elif tag == "quit":
                            break
                        elif tag == "openterminal":
                            os.system('cmd /c "start powershell.exe"')
                            print("terminal has been opened")
                        elif tag == "sleepmode":
                            os.system('cmd /k "rundll32.exe powrprof.dll,SetSuspendState"')
                        elif tag == "opennotepad":
                            os.system('cmd /c "start notepad++.exe"')
                            print("notepad++ has been started")
                        elif tag == "trainai":
                            print("AI will start to train and get back when training is done")
                            global train
                            train = True
                            break
                        else:
                            print("Somehing that shouldn't happen just happened. Quitting . . .")
                            exit(1)
                    else:
                        # print an error message when the ai wasn't confident enough
                        print("I didn't get that, try again")
                        save = False

                    # try to save new expressions in intents.json file
                    try:
                        # check wether new data should be saved or not
                        if save == True and data:
                            # check wether expression is already known in intents.json file
                            if lastSaid in data["intents"][(([x for x in range(len(data["intents"])) if data["intents"][x]["tag"] == previousTag])[0])]["patterns"]:
                                print("data already saved")
                            else:
                                # save new data
                                data["intents"][([x for x in range(len(data["intents"])) if data["intents"][x]["tag"] == previousTag])[0]]["patterns"].append(lastSaid)
                                global dataChanged
                                dataChanged = True
                        else:
                            print("data not saved")
                    except:
                        # an exception gets thrown when the program just started, but data most often is saved
                        print("Couldn't save data. Maybe the program just started")
                    
                    # update the vars
                    previousTag = tag
                    lastSaid = inp
            except:
                # speech recognition didn't understand the user acoustically
                print("I didn't hear you, try again!")
                pressed = False
    
    if dataChanged:
        # write new data to the intents.json file
        with open("intents.json", "w") as file:
            json.dump(data, file)

# start the whole process
main()

if train:
    # start training process and tell it that it should start the bot after it's done again
    os.system('cmd /k "py train.py -s"')






