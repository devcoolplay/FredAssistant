# This is the Fred assistant bot in a very early stage.
# Updates with customization process coming soon.

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import os
#from pynput import keyboard
import tflearn
import keyboard
import tensorflow
import random
import speech_recognition as sr
import json
import pickle

speech_engine = sr.Recognizer()
isTrained = True
dataChanged = False

current = set()
train = False
pressed = False

def setPressed():
    global pressed
    pressed = True

keyboard.add_hotkey('alt + s', setPressed)

with open("intents.json", "r") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    isTrained = False

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

if isTrained: 
    model.load("model.tflearn")
else:
    print("No trained model found!")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def chat():
    print("Start talking with Fred! (type \"quit\" to stop)")
    global pressed
    previousTag = "nothing"
    lastSaid = "nothing"
    while True:
        if pressed:
            inp = "nothing"
            try:
                with sr.Microphone() as micro:
                    print("Recording...")
                    audio = speech_engine.record(micro, duration=3)
                    print("Recognition...")
                    text = speech_engine.recognize_google(audio, language="de-DE")
                    print(text)

                    inp = text
                    pressed = False

                    results = model.predict([bag_of_words(inp, words)])[0]
                    results_index = numpy.argmax(results)
                    tag = labels[results_index]
                    
                    save = True

                    if results[results_index] > 0.7:
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
                        elif tag == "trainai":
                            global train
                            train = True
                            break
                        else:
                            print("I didn't get that, try again")
                            save = False

                    try:
                        if save == True and data:
                            if lastSaid in data["intents"][(([x for x in range(len(data["intents"])) if data["intents"][x]["tag"] == previousTag])[0])]["patterns"]:
                                print("data already saved")
                            else:
                                data["intents"][([x for x in range(len(data["intents"])) if data["intents"][x]["tag"] == previousTag])[0]]["patterns"].append(lastSaid)
                                global dataChanged
                                dataChanged = True
                        else:
                            print("data not saved")
                    except:
                        print("Couldn't save data. Maybe the program just started")
                    
                    previousTag = tag
                    lastSaid = inp
            except:
                print("I didn't hear you, try again!")
                pressed = False
    
    if dataChanged:
        with open("intents.json", "w") as file:
            json.dump(data, file)

chat()

if train:
    os.system('cmd /k "py train.py -s"')






