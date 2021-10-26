# This is the Fred assistant bot in a very early stage.
# Updates with customization process coming soon.

# imports
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from tqdm import tqdm
import numpy
import os
import tflearn
import random
import keyboard
import speech_recognition as sr
import json
import pyttsx3
import pickle

# progress bar
pbar = tqdm(total=100)

# initialize text-to-speech engine
tts = pyttsx3.init()
pbar.update(10)
voices = tts.getProperty("voices")
pbar.update(10)
tts.setProperty("voice", "german+f3")
pbar.update(10)
#tts.setProperty("gender", "male")
# initialize the speech recognizer engine
speech_engine = sr.Recognizer()
pbar.update(10)
# set a variable to check wether the bot is already trained
isTrained = True
# init variable to check wether the training data got changed by the bot
dataChanged = False
# init variable to check if the user wants the bot to train with its collected data 
train = False

# init variables for hotkey detection
current = set()
pressed = False

# init vars to define an exit command
executeExitCommand = False
exitCommand = "nothing"

# function for saying something
def say(text):
    tts.say(text)
    tts.runAndWait()

# set the pressed variable to true
def setPressed():
    global pressed
    pressed = True

# add a hotkey to listen to
keyboard.add_hotkey('alt + s', setPressed)

# open the intents.json file in read mode to read the data
with open("intents.json", "r") as file:
    data = json.load(file)
pbar.update(10)

# function for responding to intents
def respond(tag):
    global data
    for intent in data["intents"]:
        if intent["tag"] == tag:
            say(intent["responses"][random.randint(0, len(intent["responses"]) - 1)])
            pass


# try to open training data. if training data exists go on, and if not close the program with an error message
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    isTrained = False
pbar.update(10)

# initialize nural network
net = tflearn.input_data(shape=[None, len(training[0])])
pbar.update(5)
net = tflearn.fully_connected(net, 8)
pbar.update(5)
net = tflearn.fully_connected(net, 8)
pbar.update(5)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
pbar.update(5)
net = tflearn.regression(net)
pbar.update(5)
model = tflearn.DNN(net)
pbar.update(15)

if isTrained: 
    # load the training data
    model.load("model.tflearn")
else:
    print("No trained model found!")
pbar.close()

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
    global executeExitCommand
    global exitCommand
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
                print("1")
                with sr.Microphone() as micro:
                    print("Recording...")
                    audio = speech_engine.record(micro, duration=3)
                    print("Recognition...")
                    text = speech_engine.recognize_google(audio, language="de-DE")
                    print(text)
                    print("2")

                    inp = text
                    # reset pressed, so it won't start this process for no reason
                    pressed = False

                    # interpret what the user wants to happen
                    results = model.predict([bag_of_words(inp, words)])[0]
                    results_index = numpy.argmax(results)
                    tag = labels[results_index]
                    
                    # init a variable to tell the program wether a new expression should be saved in the intents.json file
                    save = True
                    print("result" + str(results))
                    # check wether the ai is confident enough with its interpretations
                    if results[results_index] > 0.7:
                        # execute the things that the ai understood
                        if tag == "openbrowser":
                            os.system('cmd /c "start brave.exe"')
                            print("Brave browser has been started")
                            respond(tag)
                        elif tag == "shutdown":
                            # set exit command and break out of loop
                            executeExitCommand = True
                            exitCommand = 'cmd /k "shutdown -s"'
                            respond(tag)
                            break
                        elif tag == "waswrong":
                            save = False
                            respond(tag)
                        elif tag == "openspotify":
                            respond(tag)
                            os.system('cmd /c "start spotify.exe"')
                            print("Spotify has been opened")
                        elif tag == "openfiles":
                            respond(tag)
                            os.system('cmd /c "start explorer.exe"')
                            print("explorer has been opened")
                        elif tag == "openvscode":
                            respond(tag)
                            os.system('cmd /c "code"')
                            print("visual studio code has been opened")
                        elif tag == "quit":
                            respond(tag)
                            break
                        elif tag == "openterminal":
                            respond(tag)
                            os.system('cmd /c "start powershell.exe"')
                            print("terminal has been opened")
                        elif tag == "sleepmode":
                            respond(tag)
                            executeExitCommand = True
                            exitCommand = 'cmd /k "rundll32.exe powrprof.dll,SetSuspendState"'
                        elif tag == "opennotepad":
                            respond(tag)
                            os.system('cmd /c "start notepad++.exe"')
                            print("notepad++ has been started")
                        elif tag == "trainai":
                            respond(tag)
                            print("AI will start to train and get back when training is done")
                            # set exit command and break out of loop
                            executeExitCommand = True
                            exitCommand = 'cmd /k "py train.py -s"'
                            break
                        elif tag == "sortdownloads":
                            os.system('cmd /c "cd G:\Programmierung\Python\downloadsAutomation && py main.py \"C:/Users/zink2/Downloads\""')
                            respond(tag)
                        else:
                            print("Somehing that shouldn't happen just happened. Quitting . . .")
                            exit(1)
                    else:
                        # print an error message when the ai wasn't confident enough
                        say("Sorry, Das habe ich nicht verstanden")
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
            except Exception as e:
                print(e)
                say("Ich habe dich nicht verstanden. Sage das bitte nocheinmal")
                # speech recognition didn't understand the user acoustically
                print("I didn't hear you, try again: ")
                pressed = False
    
    if dataChanged:
        # write new data to the intents.json file
        with open("intents.json", "w") as file:
            json.dump(data, file)

# start the whole process
main()


if executeExitCommand:
    executeExitCommand = False
    # execute exit command
    os.system(exitCommand)






