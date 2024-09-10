import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks import python
from mediapipe.tasks.python import BaseOptions, vision
from collections import deque
from mediapipe.framework.formats import landmark_pb2
from Support_clean import * 
CLICKDISTANCE = 0.11

'''
mp_hands is a wrapper class for mediapipe hands
handmodel maintains abstractions of the mediapipe data; 
i.e. average points in deques; bounding boxes etc




'''


class mp_hand():
    '''
    Wrapper class for hand model and mediapipe
    
    '''
    def __init__(self, landmarks, num_hands = 1):
        self.h1 = 0
        self.h2 = 0
        
        self.landmarks = landmarks
        self.num_hands = 0
        self.hands = 0
        self.results = None
        mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.draw_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=10)
        
        
        
    def initialise_hands(self, viewport_w, viewport_h, num_hands=1):
        self.num_hands = num_hands
        if num_hands == 1:
            self.h1 = handmodel(self.landmarks, viewport_h, viewport_w)
            return self.h1
        if num_hands == 2:
            print("TWO HANDS")
            self.h1 = handmodel(self.landmarks, viewport_h, viewport_w)
            self.h2 = handmodel(self.landmarks, viewport_h, viewport_w)
            return (self.h1, self.h2)
        elif num_hands > 2:
            raise ValueError("Hands number must be one or two")

    def initialise_model(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4)


    def check_image(self, frame, update_data_points = True,draw_and_return_frame = False):
        self.results = self.hands.process(frame)
        if self.results.multi_hand_landmarks and update_data_points:
            for hand_index, landmark_points in enumerate(self.results.multi_hand_landmarks):
                if draw_and_return_frame:
                    frame_to_return = self.mp_drawing.draw_landmarks(frame, landmark_points, self.mp_hands.HAND_CONNECTIONS,self.draw_spec,self.draw_spec)
                    return frame
                if self.num_hands == 1:
                    current_hand_model = self.h1
                else:
                    current_hand_model = self.h1 if hand_index == 0 else self.h2
                current_hand_model.update_values(landmark_points.landmark)
            return True
        return False

                # print(len(landmark_points.landmark))
    def get_only_current_vals_as_tuples(self, frame):
        self.results = self.hands.process(frame)
        ret = []
        if self.results.multi_hand_landmarks:
            for hand_index, landmark_points in enumerate(self.results.multi_hand_landmarks):
                for p,point in zip(landmark_points.landmark,self.landmarks):
                    ret.append((p.x, p.y, p.z))
                # print(landmark_points, hand_index)
        return ret
        
        # for hand_landmarks in results.left_hand_landmarks.landmark
        
class handmodel():
    '''
    Manages data from MP 
    i.e. boxes; distances etc
    
    '''
    def __init__(self, landmarks, viewport_h = 0, viewport_w = 0, deque_length = 3):
        self.landmarks_deque = {i: deque(maxlen=deque_length) for i in landmarks}
        print("DEQUES")
        self.average_landmark_location = {i: (0,0,0) for i in landmarks}
        self.landmarks = landmarks
        self.deque_len = deque_length
        self.viewport_h = viewport_h
        self.viewport_w = viewport_w
        

    
    def update_values(self, points):
        for p,point in zip(points,self.landmarks):
            self.landmarks_deque[point].append((p.x, p.y, p.z))
        self.compute_averages()
        

    def get_averaged_values(self, aslist = False):
        if aslist:
            return [(v1, v2, v3) for v1, v2, v3 in self.average_landmark_location.values()]
        return self.average_landmark_location
    
    def get_bounding_box(self, img_width, img_height, buffer_ratio = 0.10):
        # No need to track individual x0, y0, x1, y1
        # print("asking for bbox")
        # print(self.average_landmark_location)
        
        # print(min([value[0] for value in self.average_landmark_location.values()]))
        # print(min([value[1] for value in self.average_landmark_location.values()]))
        buffer = img_width * buffer_ratio  # calculate buffer as 10% of image width
        x_min = min(value[0] for value in self.average_landmark_location.values()) * img_width - buffer
        y_min = min(value[1] for value in self.average_landmark_location.values()) * img_height - buffer
        x_max = max(value[0] for value in self.average_landmark_location.values()) * img_width + buffer
        y_max = max(value[1] for value in self.average_landmark_location.values()) * img_height + buffer
        return ((int(x_min), int(y_min)),(int(x_max), int(y_max)))

    def get_bounding_box_aspect_ratio(self, img_width, img_height, buffer_ratio = 0.07, aspect_ratio = 1.0):
        buffer = img_width * buffer_ratio
        x_min = min(value[0] for value in self.average_landmark_location.values()) * img_width - buffer
        y_min = min(value[1] for value in self.average_landmark_location.values()) * img_height - buffer
        x_max = max(value[0] for value in self.average_landmark_location.values()) * img_width + buffer
        y_max = max(value[1] for value in self.average_landmark_location.values()) * img_height + buffer

        # Calculate the current width and height of the bounding box
        width = x_max - x_min
        height = y_max - y_min

        # If the current aspect ratio is less than the desired aspect ratio
        if width / height < aspect_ratio:
            # Increase the width
            width = height * aspect_ratio
            x_min = (x_max + x_min - width) / 2
            x_max = x_min + width
        else:
            # Increase the height
            height = width / aspect_ratio
            y_min = (y_max + y_min - height) / 2
            y_max = y_min + height

        return ((int(x_min), int(y_min)),(int(x_max), int(y_max)))
        
    def compute_averages(self, points = None): 
        if points is None:
            points = self.landmarks
        for lm in points: 
            data = self.get_single_landmark(lm)
            ax,ay,az = 0,0,0
            for d in data:
                ax += d[0]
                ay += d[1]
                az += d[2]
            ay = ay/len(data)
            ax = ax/len(data)
            az = az/len(data)
            self.average_landmark_location[lm] = (ax,ay,az)
            
        
    def pinch_fingers_touching(self):
        # f1 = self.get_single_landmark('THUMB_TIP', average=True)
        # f2 = self.get_single_landmark('INDEX_FINGER_TIP', average=True)
        
        f1 = self.landmarks_deque['THUMB_TIP'][-1]
        f2 = self.landmarks_deque['INDEX_FINGER_TIP'][-1]
        print("F1 F2" , f1 , f2)
        d = distance_between_points(f1, f2)
        print(d)
        if d < CLICKDISTANCE: #MAGIC VALUE FOR MY HANDS ONLY IM AFRAID
            return True
        return False
        return d

    
    def get_single_landmark(self, landmark_name,scale_to_display = False, average = False):

        if average:
            a =  self.average_landmark_location[landmark_name]
        else:
            a = self.landmarks_deque[landmark_name]
 
        return a
    def get_pinch_points(self):
        a = self.get_single_landmark('THUMB_TIP')
        b = self.get_single_landmark('INDEX_FINGER_TIP')
        return (a,b)
    
    
    def get_scroll_tips(self):
        a = self.get_single_landmark('MIDDLE_FINGER_TIP')
        b = self.get_single_landmark('INDEX_FINGER_TIP')
        return (b,a)
    def get_3d_area_between_landmarks(self, landmark_name):
        pass
        
    def print_landmarks_to_console(self, landmark_name):
        for a in self.landmarks_deque[landmark_name]:
            print(a)
    def print_all(self):
        for a,b in zip(self.average_landmark_location, self.landmarks_deque):
            print(a, self.average_landmark_location[a])
            print(b, self.landmarks_deque[b])
            
            
            