import cv2
import numpy as np

def extract_hand_contour(image, landmarks):
    # Create a blank mask
    mask = np.zeros_like(image, dtype=np.uint8)
    

    # Convert the landmarks to an array of points
    points = np.array(landmarks, dtype=np.int32)

    # Draw contours around the landmarks
    cv2.drawContours(image, [points], -1, 255, thickness=cv2.FILLED)
    return image
    


    # Extract the hand region using the mask
    hand_region = cv2.bitwise_and(image, image, mask=mask)

    return hand_region



import numpy as np
import cv2

def threshold_image(frame):  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    
    
    return thresh

    

def threshold_image(frame, colour = (255, 255, 0)):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    blur = cv2.GaussianBlur(gray, (5, 5), 0)


    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    colored_frame = np.zeros_like(frame)
    colored_frame[:] = colour  # 

    hand_mask = cv2.bitwise_not(thresh)


    hand_region = cv2.bitwise_and(frame, frame, mask=hand_mask)


    background_mask = thresh
    colored_frame = cv2.bitwise_and(colored_frame, colored_frame, mask=background_mask)

    result = cv2.add(colored_frame, hand_region)

    return result

