# Real time gesture interface using RGB cameras and machine learning.

Broadly, the project contains three parts:

* A wrapper for mediapipe for hand detection

* A custom TF keras NN for classifying gestures

* Boilerplate for use with some HCI interface, basic MacOS interface included.

Models are included for the below data; their efficacy is available in report.pdf



### Real-time responsive input gestures using a standard rgb camera for computer systems.

This project was primarily developed on and for use on apple MacOS it *should* work with microsoft windows environments but it is not tested.

Code for the windows interface was therefore simply generated.



Models are located in /Models

Training tools are in /Training

run 'run.py' (no args required, args set as flags in code)

Depending on your macos security settings, you may need to enable some permissions when prompted.

Turn OFF any apple silicon powered webcam filters / portrait mdoe etc.


MOST getstures should work on your version of macos and windows, linux is absolutely not guaranteed!



gestures = {
    0: "pinch",     
    1: "point",
    2: "idle",
    3: "2-finger",
    4: "spread"
}




## Training
The following is intended for training input directly from the webcam and is, as such, rather rudimentary.

You will need to specify in the code the .csv filename you'd like to make.
Run trainer.py.

Create the gesture you'd like for the csv. i.e. 'spread.csv' and press the '4' key.

Run prerpocessor.py and specifiy in code the csv path and the output category number



There are models pretrained in models, the trainer pytohn file in trainers will help to create more data (currently set to waggle your hands in front of the webcam mode).

Specify a csv within trainer.

Run preprocessor to tidy up and specify number of lines and output label in there too.

Run train_model.py to generate a model with all. By default it takes 20% of data to train on.


Run train_model.py to create a model file in the same directory (name it in code)

point 'recognizer.py to that model


## Other structure:


run.py is main entry point.

Image_methods contains the code for cuttng out the hand. Yes, I did cherry pick lighting conditions!

Handmodel contains abstractions of mediapipe hands and a handler class for hand data

statemachine.py is as it seems



NOTE:

interface.py is ONLY really implemented for macos. Disable the flag for 'USE INTERFACE' in run.py in order to skip it (it's not particularly well implemented anyway.)


That's about it. Use at your own peril future students!