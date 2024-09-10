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
# from extract_hand import *
from Support_clean import *
from recognizer import *


interact = interaction_model()
DISPLAY_WIDTH = interact.display_w
DISPLAY_HEIGHT = interact.display_h
IMAGE_INPUT_WIDTH = 640
IMAGE_INPUT_HEIGHT = 480
CLICK_THRESHOLD = 0.115






'''
Use the following flag to turn actually interfacing with the system on and off

false = only gesture recognizer & image methods
true = gesture recognzier, image methods and pyautogui calls
'''
USE_INTERFACE = True


#Open Webcam
cap = cv2.VideoCapture(0)

#Hand Drawing (if flag)
mp_drawing = mp.solutions.drawing_utils
draw_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
DRAW_HANDS = False
SHOW_FEED = True

#Gesture recognizer (TF)
gesture_recognizer = handGestureRecognizer()
gesture_recognizer.initialise()

#interface 
interface = interaction_model()

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



#StateMachine local
statemachine = StateMachine( initial= "idle")




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

frame_counter = 0



#For analysis
if False:
    '''
    average the gesture accuracy
    '''
    num_gesture_checks = 0
    gesture_totals = [0] * 5
    
if False:
    number_correct = 0
    number_incorrect = 0
    intended_gesture = "pinch"
    
    
gestures = {
    0: "pinch",
    1: "point",
    2: "idle",
    3: "2-finger",
    4: "spread"
}


colours = {
    'pinch': (64, 64, 255),
    'point': (255, 128, 0),
    'idle': (0, 255, 128),
    '2-finger': (0, 255, 255),
    'spread': (128, 0, 128),
    'pinch_w_click': (200, 123, 121),
    'pinch_no_click': (88, 101, 103),
    '': (0,0,0)
}



while 1:
    gtext = ""
    current_gesture = ""
    prev_gesture = ""
    hands_detected = False
    
    
    #WStaggered polling of each stage, asusming 30fps run rate. High doesn't hurt!
    do_averages = frame_counter%10 == 0
    do_mp_hand = frame_counter%1 == 0
    do_crop_to_hands = True
    do_gesture_recoginze = frame_counter%1 == 0
    do_redraw_pipe = frame_counter%1 == 0 #note that this defines the fps 
    frame_counter += 1
    do_update_statemachine = frame_counter%1 == 0
    state = ""
    
    if frame_counter >= 30:
        frame_counter = 0
    
    state = statemachine.current_state()
    
    #cv2 funcs
    ret, image = cap.read()
    key = cv2.waitKey(1)
    processed_image = image
    cropped_image = image
    raw = image

    
    #resize for performance
    image = cv2.resize(image, (IMAGE_INPUT_WIDTH, IMAGE_INPUT_HEIGHT))
    
    #setup for mediapipe
    image = cv2.flip(image, 1)
    bgr2rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    #check with hand model for hand
    if do_mp_hand:
        hands_detected = mp_hand_processor.check_image(bgr2rgb)
        
    #get gesture
    if do_gesture_recoginze and hands_detected:
        # print("DO GESTURE REC")
        angles_vec = compute_angles(h1.get_averaged_values(aslist = True))
        probs = gesture_recognizer.predict(angles_vec)
        prob_dec = np.argmax(probs)
        current_gesture = gestures[prob_dec]
    if do_update_statemachine:
        state = statemachine.current_state()
        
        if current_gesture == 'pinch':
            if h1.pinch_fingers_touching():
                current_gesture = 'pinch_w_click'
            else:
                current_gesture = 'pinch_no_click'
        statemachine.transition(current_gesture)
        
    
    print(current_gesture)
    if USE_INTERFACE:
        
        if state == 'idle':
            interface.stop_mouse()
        elif state == 'point':
            pass
        elif state == 'point':
            pass
        elif state == 'track_location_no_click':
            a = h1.get_single_landmark('INDEX_FINGER_TIP')
            interface.disengage_click()
            xv,yv = (compute_velocity_from_deque(a))
            interface.move_mouse(xv,yv,scale = 200)
        elif state == 'track_location_click':
            a = h1.get_single_landmark('INDEX_FINGER_TIP')
            xv,yv = (compute_velocity_from_deque(a))
            interface.engage_click()
            interface.move_mouse(xv,yv,scale = 200)
            pass
        elif state == 'pinch_zoom':
            
            p1,p2 = h1.get_pinch_points()
            
            
            pass
        elif state == 'spread':
            interface.spread()
            pass
        elif state == 'pan':
            pass
                
            
    

    
    #Crop the cv2 image to be only the users hahd as defined in the bounding box.
    if do_crop_to_hands and hands_detected:
        bbox = h1.get_bounding_box(image.shape[1], image.shape[0])  # get bounding box # green color
        color = (0, 255, 0)  # green color
        cv2.rectangle(image, bbox[0], bbox[1], color, thickness)
        #one bounding box is the same aspect ratio; one is the bounding box
        bbox2 = h1.get_bounding_box_aspect_ratio(image.shape[1], image.shape[0])  # get bounding box
        color = (255, 255, 0)  # green color
        cv2.rectangle(image, bbox2[0], bbox2[1], color, thickness)

        if bbox2[0][0] >= 0 and bbox2[0][1] >= 0 and bbox2[1][0] <= image.shape[1] and bbox2[1][1] <= image.shape[0]:
            image = cv2.resize(image[bbox2[0][1]:bbox2[1][1], bbox2[0][0]:bbox2[1][0]], image.shape[:2][::-1])
            cropped_image = image
            image = threshold_image(image, colours[current_gesture])
            # processed_image = threshold_image(image)
            
            
          
    
    #put output feedback on the display (each task shown on it's relevant frame)
    text = (f"{str(frame_counter)} -- {'a' if do_averages else ' '},{'h' if do_mp_hand else ' '},{'g' if do_gesture_recoginze else ' '},{'rd' if do_redraw_pipe else ' '} {current_gesture } {str(gtext)}")
  
  
  
  
    if key == (ord('4')):
        h1.print_all()
    if key == (ord('q')):
            # print("Correct", number_correct, "and incorrect", number_incorrect)
            break

    
    cv2.putText(image, text, position_fps_counter, font, font_scale, color, thickness)
    final = cv2.resize(image, (1280 , 720))
    cv2.imshow("SuperHands", final)
    
    
    
    '''
    Uncomment the following for the demo video effect
    '''
    # cv2.imshow('test', cropped_image)
    # cv2.imshow('raw', raw)
    
    
  

cap.release()
cv2.destroyAllWindows()