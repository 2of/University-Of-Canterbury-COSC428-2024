import numpy as np
import json

from collections import deque



def midpoint(p1, p2):
    ''' 

    2p vec return ave 

    '''
    x = np.array((int(p1[0] + p2[0])/2, int(p1[1] + p2[1])/2))
    return x.astype(int)



import math

def distance_between_points(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    return distance


class SmoothMouse:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = []

    def add_point(self, x, y):
        self.buffer.append((x, y))
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def get_average(self):
        if not self.buffer:
            return None

        # Calculate average coordinates
        avg_x = sum(x for x, _ in self.buffer) / len(self.buffer)
        avg_y = sum(y for _, y in self.buffer) / len(self.buffer)
        return int(avg_x), int(avg_y)
    
    



class handmodel():
    def __init__(self, img_width=1, img_height =1 , points_of_interest = ["MIDDLE_FINGER_TIP","INDEX_FINGER_TIP","RING_FINGER_TIP","PINKY_TIP","THUMB_TIP"]): 
        # self.points_of_interest = {p: np.zeros(3) for p in points_of_interest}
        self.points_of_interest = {p: np.zeros(3).tolist() for p in points_of_interest}
        self.current_gesture = "NONE"
        self.x0 = 3840
        self.y0 = 3840
        self.x1 = float('inf')  # Initialize to a large positive number
        self.y1 = float('inf')  # Initialize to a large positive number
        self.img_width = img_width
        self.img_height = img_height
        self.points_of_interest_dq = {p: deque(maxlen=8) for p in points_of_interest}
        
        

    def clear_bounding_box(self):

        self.box_updated = 0
    def update_img_size(self,img_width,img_height):
        self.img_width = img_width
        self.img_height = img_height
        
        
        
    def is_pointing_index(self):

        distances = [np.linalg.norm(np.array(self.points_of_interest["INDEX_FINGER_TIP"]) - np.array(self.points_of_interest[p])) for p in self.points_of_interest if p != "INDEX_FINGER_TIP"]
        print(distances)

        if max(distances) == np.linalg.norm(np.array(self.points_of_interest["INDEX_FINGER_TIP"]) - np.array(self.points_of_interest["THUMB_TIP"])):
            return True
        else:
            return False
    def is_pointing(self):
        
        
        
        
    
        pointing = False

        index_tip = self.points_of_interest["INDEX_FINGER_TIP"]
        thumb_tip = self.points_of_interest["THUMB_TIP"]
        middle_tip = self.points_of_interest["MIDDLE_FINGER_TIP"]
        # Calculate direction 
        direction_vec = np.subtract(thumb_tip, index_tip)

        # Calculate normalized 
        norm_direction_vec = direction_vec / np.linalg.norm(direction_vec)

        # Calculate angle 
        if middle_tip is not None:
            middle_vec = np.subtract(middle_tip, index_tip)
            angle = np.arccos(np.dot(norm_direction_vec, middle_vec) /
                                (np.linalg.norm(norm_direction_vec) * np.linalg.norm(middle_vec)))
        else:
            angle = 0  


        distance = np.linalg.norm(direction_vec)

      
        pointing_angle_threshold = np.pi / 3  
        pointing_distance_threshold = self.img_width * 0.2  

        # Check if pointing criteria are met
        if (angle <= pointing_angle_threshold and
                distance >= pointing_distance_threshold):
            pointing = True
       
        return pointing


    def get_average_point(self, point):
        if len(self.points_of_interest[point]) == 0:
            return None
        
        
        avg_point = np.mean(self.points_of_interest_dq[point], axis=0)
        # print("average", avg_point, "vs", self.get_point(point))
        return avg_point.tolist()
    
    
    

        
    def update_point2(self, point, x, y, z):
        self.points_of_interest[point][0] =  x
        self.points_of_interest[point][1] =  y
        self.points_of_interest[point][2] =  z
        # self.x0 = max(x, self.x0)
        # self.x1 = min (x, self.x1)
        
        # self.y0 = max(y, self.y0)
        # self.y1  = min(y, self.y1)
        
    def update_point(self, point, x, y, z):
        self.points_of_interest[point][0] =  x
        self.points_of_interest[point][1] =  y
        self.points_of_interest[point][2] =  z
        self.points_of_interest_dq[point].append([x, y, z])

    def get_bounding_box(self, buffer = 48):
            # No need to track individual x0, y0, x1, y1
            x_min = min(value[0] for value in self.points_of_interest.values())
            y_min = min(value[1] for value in self.points_of_interest.values())
            x_max = max(value[0] for value in self.points_of_interest.values())
            y_max = max(value[1] for value in self.points_of_interest.values())
            # buffer = min()
            # Ensure coordinates are within image bounds
            x_min = int(max(0, min(x_min, self.img_width - 1))) - buffer
            y_min = int(max(0, min(y_min, self.img_height - 1))) - buffer
            x_max = int(max(0, min(x_max, self.img_width - 1)))  + buffer
            y_max = int(max(0, min(y_max, self.img_height - 1))) + buffer

            return ((x_min, y_min),(x_max, y_max))
        
    def normalize_points_within_bounding_box(self):
        # Calculate bounding box coordinates (Xmin, Ymin, w, h) based on your points_of_interest
        x_min = min(value[0] for value in self.points_of_interest.values())
        y_min = min(value[1] for value in self.points_of_interest.values())
        x_max = max(value[0] for value in self.points_of_interest.values())
        y_max = max(value[1] for value in self.points_of_interest.values())
        w = x_max - x_min
        h = y_max - y_min

        # Normalize each point within the bounding box
        normalized_points = {}
        for point, (x, y, z) in self.points_of_interest.items():
            normalized_x = (x - x_min) / w
            normalized_y = (y - y_min) / h
            normalized_points[point] = {"x": normalized_x, "y": normalized_y, "z": z}

        return normalized_points
    
    
    def get_landmarks_list(self):
        return self.points_of_interest
    def get_points_as_list_of_tuples(self):
        return [(int(value[0]), int(value[1])) for value in self.points_of_interest.values()]


    def get_point(self, point):
        return (self.points_of_interest[point][0],
                self.points_of_interest[point][1],
                self.points_of_interest[point][2])

    def get_highest_point(self):
        return  max(value[1] for value in self.points_of_interest.values())
    
    
    def get_points(self):
        return json.dumps(self.points_of_interest)
    
    
    def get_points_with_title(self, title):
        result = self.get_points()
        return json.dumps({"gesture_name": title, "result": result})
    
   
    def get_ave_line(self, keys):
        """
        Computes a straight line passing through the middle of specified hand landmarks.

        Args:
            keys (list of str): List of keys corresponding to hand landmarks.

        Returns:
            tuple: A tuple of two (x, y) points representing the line.
        """
        num_landmarks = len(keys)
        if num_landmarks == 0:
            raise ValueError("No landmarks provided.")

        # Compute the average position of the specified landmarks
        central_point = np.mean([self.get_average_point(key) for key in keys if self.get_average_point(key) is not None], axis=0)

        left_point = (self.get_average_point(keys[0])[0], central_point[1])
        right_point = (100, central_point[1])

        return left_point, right_point

    def pointing_at_screen_point(self, screen_width=1920, screen_height=1080, assumed_distance=100):
  

    
        palm_center = np.array(self.get_average_point("INDEX_FINGER_DIP"))  
        index_tip = np.array(self.get_average_point("INDEX_FINGER_TIP"))


        if not all(np.any(value) for value in [palm_center, index_tip]):
            return None


        direction_vector = index_tip - palm_center

        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            direction_vector /= norm
        else:
            print("breaki pointing")
            return None

        # Calculate the direction vector
        direction_vector = index_tip - palm_center

        # Normalize the direction vector (handle division by zero)
        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            direction_vector /= norm
        else:
            print("Point vec has no magnitude! Break!.")
            return None


        scaled_direction_vector = direction_vector * assumed_distance

        screen_point = (
            int(palm_center[0] + scaled_direction_vector[0]),
            int(palm_center[1] + scaled_direction_vector[1]),
        )

        # Ensure coordinates are within screen bounds
        screen_point = (
            max(0, min(screen_point[0], screen_width - 1)),
            max(0, min(screen_point[1], screen_height - 1)),
        )

        return screen_point

    
class FileAppender:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, 'a+')

    def append(self, data):
        print(data)
        self.file.write(data)

    def close(self):
        self.file.close()



class state_machine_gesture_recognition():
    def __init__(self):
        self.state = 'IDLE'
        
        
    def hand_detected():
        pass
    def no_hand_detected():
        pass
    def fingers_open():
        pass
    def quick_click():
        pass
    def two_hand_detected():
        pass
    
    
 