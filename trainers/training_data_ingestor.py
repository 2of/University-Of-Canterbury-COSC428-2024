#libs
from subprocess import call
from Support_funcs import *
from Image_Methods import * 
import numpy as np
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks import python
from mediapipe.tasks.python import BaseOptions, vision
#locals
from interface import *
from statemachine import *
from handModel import *
from extract_hand import *
import csv
from Support_clean import *
'''
Very rudimentary file to create csv data of angles between keypoints. 

4 = write row

Will begin a cv2 stream from webcam. press 4 to save the relative angle row to the .csv you specify


'''
interact = interaction_model()
DISPLAY_WIDTH = interact.display_w
DISPLAY_HEIGHT = interact.display_h
IMAGE_INPUT_WIDTH = 640
IMAGE_INPUT_HEIGHT = 480
#StateMachine local
statemachine = StateMachine()

#MediaPipe Hands Initialization 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.4)

#Open Webcam
cap = cv2.VideoCapture(0)

#Hand Drawing (if flag)
mp_drawing = mp.solutions.drawing_utils
draw_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
DRAW_HANDS = False
SHOW_FEED = True


#Gesture Model
# base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
# options = vision.GestureRecognizerOptions(base_options=base_options)
# recognizer = vision.GestureRecognizer.create_from_options(options)


#We consider ALL points on the hand (for now)
notable_landmarks = [
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_TIP"
]






# Right and Left Hands 
# left_hand_data  = handmodel(points_of_interest=notable_landmarks)
# right_hand_data = handmodel(points_of_interest=notable_landmarks)
mp_hand_processor = mp_hand(notable_landmarks)
h1,h2 = mp_hand_processor.initialise_hands(IMAGE_INPUT_WIDTH, IMAGE_INPUT_HEIGHT,num_hands=2)
mp_hand_processor.initialise_model()

# Fonts
position_fps_counter = (10, 30) 
font = cv2.FONT_HERSHEY_SIMPLEX 
font_scale = 1  
color = (255, 0, 255) 
thickness = 2  

approx_distance_from_camera = 100 #units?
frame_counter = 0
filename = "idle.csv"
with open(filename, mode="w", newline="") as csvfile:
    
    csvwriter = csv.writer(csvfile)

    while 1:
        
        #WHEN TO TRIGGER EACH CHECK FOR COMPUTE EFFICIECNY
        do_averages = frame_counter%10 == 0
        do_mp_hand = frame_counter%1 == 0
        do_crop_to_hands = False
        do_gesture_recoginze = frame_counter%2 == 0
        do_redraw_pipe = frame_counter%1 == 0 #note that this defines the fps so ill need to change this
        frame_counter += 1
        if frame_counter >= 30:
            frame_counter = 0

        
        state = statemachine.current_state()
        #cv2 funcs
        ret, image = cap.read()
        key = cv2.waitKey(1)
        #resize for performance

        image = cv2.resize(image, (IMAGE_INPUT_WIDTH, IMAGE_INPUT_HEIGHT))
        
        #setup for mediapipe
        image = cv2.flip(image, 1)
        bgr2rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #check with hand model
        if do_mp_hand:
            print("Check for hand")
            a = mp_hand_processor.get_only_current_vals_as_tuples(bgr2rgb)
            if a is not None:
                print("HAND DETECTED")
                b = compute_angles(a)
            if (a):
                do_crop_to_hands = True
        bbox = h1.get_bounding_box_aspect_ratio(1,1,1,1)

            # do_crop_to_hands = mp_hand_processor.check_image(bgr2rgb)
        #get gesture
        
        
        #get just box

        
        if do_crop_to_hands:
            
            # h2.compute_averages(notable_landmarks)
            # bbox = h2.get_bounding_box(image.shape[1], image.shape[0])  # get bounding box # green color
            # color = (0, 255, 0)  # green color
            # cv2.rectangle(image, bbox[0], bbox[1], color, thickness)
            # h2.compute_averages(notable_landmarks)
            # h1.compute_averages(notable_landmarks)
            bbox = h1.get_bounding_box(image.shape[1], image.shape[0])  # get bounding box # green color
            color = (0, 255, 0)  # green color
            cv2.rectangle(image, bbox[0], bbox[1], color, thickness)
            
            bbox2 = h1.get_bounding_box_aspect_ratio(image.shape[1], image.shape[0])  # get bounding box
            color = (255, 255, 0)  # green color
            cv2.rectangle(image, bbox2[0], bbox2[1], color, thickness)
            # cropped_image = image[bbox2[0][1]:bbox2[1][1], bbox2[0][0]:bbox2[1][0]]
            print(image.shape[0], image.shape[1])
            if bbox2[0][0] >= 0 and bbox2[0][1] >= 0 and bbox2[1][0] <= image.shape[1] and bbox2[1][1] <= image.shape[0]:
                image = cv2.resize(image[bbox2[0][1]:bbox2[1][1], bbox2[0][0]:bbox2[1][0]], image.shape[:2][::-1])
        text = (f"{str(frame_counter)} -- {'a' if do_averages else ' '},{'h' if do_mp_hand else ' '},{'g' if do_gesture_recoginze else ' '},{'rd' if do_redraw_pipe else ' '}")
        #overrides
        if key == (ord('2')):
            print(h2.get_averaged_values())
        if key == (ord('3')):
            h1.compute_averages(notable_landmarks)
        if key == (ord('4')):
            csvwriter.writerow(b)
            print("WROTE LINE FOR " , filename, len(b))

        
        cv2.putText(image, text, position_fps_counter, font, font_scale, color, thickness)
        final = cv2.resize(image, (1280 , 720))
        
        cv2.imshow("SuperHands", final)
    cap.release()