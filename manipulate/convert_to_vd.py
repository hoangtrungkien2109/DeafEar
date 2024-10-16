import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def draw_landmarks(image, frame_landmarks):
    pose_landmarks = frame_landmarks[:33]
    for lm in pose_landmarks:
        if not np.isnan(lm[0]) and not np.isnan(lm[1]) and not np.isnan(lm[2]):  # Check for NaN values
            x, y = int(lm[0] * image.shape[1]), int(lm[1] * image.shape[0])
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

    right_hand_landmarks = frame_landmarks[33:33 + 21]
    right_hand_present = False
    for lm in right_hand_landmarks:
        if not np.isnan(lm[0]) and not np.isnan(lm[1]) and not np.isnan(lm[2]):  # Check for NaN values
            x, y = int(lm[0] * image.shape[1]), int(lm[1] * image.shape[0])
            cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
            right_hand_present = True

    left_hand_landmarks = frame_landmarks[33 + 21:]
    left_hand_present = False
    for lm in left_hand_landmarks:
        if not np.isnan(lm[0]) and not np.isnan(lm[1]) and not np.isnan(lm[2]):  # Check for NaN values
            x, y = int(lm[0] * image.shape[1]), int(lm[1] * image.shape[0])
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
            left_hand_present = True

    if not right_hand_present and not left_hand_present:
        return None  # Return None if no hands are detected

    return image



def load_and_concatenate_npy_files(npy_files):
    all_landmarks = []
    for npy_file in npy_files:
        landmarks_data = np.load(npy_file)
        all_landmarks.append(landmarks_data)

    concatenated_landmarks = np.concatenate(all_landmarks, axis=0)
    return concatenated_landmarks

def is_similar_frame(frame1, frame2, threshold=0.20):
    distance = np.linalg.norm(frame1 - frame2)
    return distance < threshold

# Get the list of .npy files to load
npy_files = ['landmarks_data1.npy',
             'landmarks_data2.npy',
             'landmarks_data3.npy',
             'landmarks_data4.npy']
concatenated_landmarks_array = load_and_concatenate_npy_files(npy_files)

frame_index = 0
num_frames = len(concatenated_landmarks_array)

image_height, image_width = 720, 1280

last_frame_landmarks = None

while frame_index < num_frames:
    image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
    frame_landmarks = concatenated_landmarks_array[frame_index]

    result_image = draw_landmarks(image, frame_landmarks)
    if result_image is None:
        frame_index += 1
        continue

    cv2.imshow('Landmarks Visualization', result_image)
    frame_index += 1

    if cv2.waitKey(20) & 0xFF == ord('q'):  
        break


# Clean up
cv2.destroyAllWindows()
