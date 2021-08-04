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

##############   ToDo   ###############
# let the bot recognize what the user wants to do
# include speech  recognition

speech_engine = sr.Recognizer()
isTrained = True

current = set()
pressed = False

def setPressed():
    global pressed
    pressed = True
    print("Hotkey works")

keyboard.add_hotkey('alt + s', setPressed)

with open("intents.json") as file:
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
            except:
                print("something went wrong during speech recognition")

            #inp = input("You: ")
            #if inp.lower() == "quit":
            #    print("Fred: Bye!")
            #    break


            results = model.predict([bag_of_words(inp, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]
            
            if results[results_index] > 0.7:
                if tag == "openbrowser":
                    os.system('cmd /c "start brave.exe"')
                    print("Brave browser has been started")
                elif tag == "openspotify":
                    os.system('cmd /c "start spotify.exe"')
                    print("Spotify has been opened")
                elif tag == "openfiles":
                    os.system('cmd /c "start explorer.exe"')
                    print("explorer has been opened")
                elif tag == "openvscode":
                    os.system('cmd /c "code"')
                    print("visual studio code has been opened")
                elif tag == "openterminal":
                    os.system('cmd /c "start powershell.exe"')
                    print("terminal has been opened")
                else:
                    print("I didn't get that, try again")

chat()







