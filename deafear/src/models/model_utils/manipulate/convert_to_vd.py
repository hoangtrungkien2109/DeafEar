import cv2
import numpy as np
from model import load_model, predict
# Correct connections for pose landmarks in MediaPipe (total 33 landmarks)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),          # Right eye to ear
    (0, 4), (4, 5), (5, 6), (6, 8),          # Left eye to ear
    (9, 10), (11, 12),                       # Shoulders
    (11, 13), (13, 15), (15, 17),            # Left arm
    (12, 14), (14, 16), (16, 18),            # Right arm
    (23, 24),                                # Hips
    (24, 26), (26, 28), (28, 32),            # Right leg
    (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (5, 6), (6, 7), (7, 8),          # Index finger
    (9, 10), (10, 11), (11, 12),     # Middle finger
    (13, 14), (14, 15), (15, 16),    # Ring finger
    (17, 18), (18, 19), (19, 20)     # Pinky finger
]

def draw_landmarks(image, frame_landmarks,line_thickness=2):
    pose_landmarks = frame_landmarks[:33]
    for lm in pose_landmarks:
        if not np.isnan(lm[0]) and not np.isnan(lm[1]) and not np.isnan(lm[2]):  # Check for NaN values
            x, y = int(lm[0] * image.shape[1]), int(lm[1] * image.shape[0])
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
            
    # Draw lines between connected pose landmarks
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = pose_landmarks[start_idx]
        end_point = pose_landmarks[end_idx]
        if (start_point[0] != 0 and start_point[1] != 0) and (end_point[0] != 0 and end_point[1] != 0):
            start_x, start_y = int(start_point[0] * image.shape[1]), int(start_point[1] * image.shape[0])
            end_x, end_y = int(end_point[0] * image.shape[1]), int(end_point[1] * image.shape[0])
            cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), line_thickness)

            
    right_hand_landmarks = frame_landmarks[33:33 + 21]
    right_hand_present = False
    for lm in right_hand_landmarks:
        if not np.isnan(lm[0]) and not np.isnan(lm[1]) and not np.isnan(lm[2]):  # Check for NaN values
            x, y = int(lm[0] * image.shape[1]), int(lm[1] * image.shape[0])
            cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
            right_hand_present = True
            
    # Draw lines between connected right hand landmarks
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = right_hand_landmarks[start_idx]
        end_point = right_hand_landmarks[end_idx]
        if (start_point[0] != 0 and start_point[1] != 0) and (end_point[0] != 0 and end_point[1] != 0):
            start_x, start_y = int(start_point[0] * image.shape[1]), int(start_point[1] * image.shape[0])
            end_x, end_y = int(end_point[0] * image.shape[1]), int(end_point[1] * image.shape[0])
            cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), line_thickness)
            
    left_hand_landmarks = frame_landmarks[33 + 21:]
    left_hand_present = False
    for lm in left_hand_landmarks:
        if not np.isnan(lm[0]) and not np.isnan(lm[1]) and not np.isnan(lm[2]):  # Check for NaN values
            x, y = int(lm[0] * image.shape[1]), int(lm[1] * image.shape[0])
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
            left_hand_present = True
            
    # Draw lines between connected left hand landmarks
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = left_hand_landmarks[start_idx]
        end_point = left_hand_landmarks[end_idx]
        if (start_point[0] != 0 and start_point[1] != 0) and (end_point[0] != 0 and end_point[1] != 0):
            start_x, start_y = int(start_point[0] * image.shape[1]), int(start_point[1] * image.shape[0])
            end_x, end_y = int(end_point[0] * image.shape[1]), int(end_point[1] * image.shape[0])
            cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), line_thickness)
             
    if not right_hand_present and not left_hand_present:
        return None  # Return None if no hands are detected

    return image



def load_and_concatenate_npy_files(model,npy_files):
    all_landmarks = []
    prediction = []
    for npy_file in npy_files:
        landmarks_data = np.load(npy_file)
        
        all_landmarks.append(landmarks_data)

        p = predict(model,landmarks_data)
        prediction.append(p)

    
    concatenated_landmarks = np.concatenate(all_landmarks, axis=0)
    concatenated_prediction = np.concatenate(prediction, axis=0)
    print("c1",concatenated_landmarks.shape)
    print("c2",concatenated_prediction.shape)
    return concatenated_landmarks, concatenated_prediction

def is_similar_frame(frame1, frame2, threshold=0.05):
    if frame1 is None:
        return False
    distance = np.linalg.norm(frame1 - frame2)
    return distance < threshold

def defineSE(arr):
    s,e = 0, len(arr)-1
    for i in range(len(arr)-1):
        if arr[i] != arr[i+1]:
            s = i + 1
            break
            
    for i in range(len(arr)-1,0,-1):
        if arr[i] != arr[i-1]:
            e = i
            break
    return s,e
    

# Get the list of .npy files to load
import glob
import os
 # load model
model = load_model("model_final.pth")

npy_folder = './temp'
npy_files = glob.glob(os.path.join(npy_folder, '*'))

concatenated_landmarks_array, concatenated_prediction_array = load_and_concatenate_npy_files(model, npy_files)

frame_index = 0
num_frames = len(concatenated_landmarks_array)

image_height, image_width = 720, 1280

start, end = defineSE(concatenated_prediction_array[:,0])
print("start", start)
print("end", end)
last_frame_landmarks = None
while frame_index < num_frames:
    image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
    frame_landmarks = concatenated_landmarks_array[frame_index]

    result_image = draw_landmarks(image, frame_landmarks)
    if (frame_index >= start and frame_index <= end) and concatenated_prediction_array[frame_index][0] == 0:
        frame_index += 1
        continue
    
    if result_image is None or is_similar_frame(last_frame_landmarks, frame_landmarks):
        frame_index += 1
        continue
    
    last_frame_landmarks = frame_landmarks

    cv2.imshow('Landmarks Visualization', result_image)
    frame_index += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break


# Clean up
cv2.destroyAllWindows()
