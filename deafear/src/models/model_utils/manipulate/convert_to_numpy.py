import cv2
# import mediapipe as mp
import numpy as np
import glob
import os
import logging
import colorlog

def file_exists(folder_path, file_name):
    file_name = f"landmarks_{file_name}.npy"
    file_path = os.path.join(folder_path, file_name)
    return os.path.isfile(file_path)

logger = logging.getLogger(__name__)
logging.basicConfig(filename='convert_data.log',level=logging.DEBUG)

video_folder = '/Users/trHien/PycharmProjects/ScrapeASLData/data'
destiny_folder = "./data_processed"

mp_holistic = mp.solutions.holistic

video_files = glob.glob(os.path.join(video_folder, '*[0-9].mp4'))[:200]

for video_file in video_files:
    file_name_with_ext = os.path.basename(video_file)
    
    file_name = os.path.splitext(file_name_with_ext)[0]
    if file_exists(destiny_folder,file_name):
        logger.warning(f"{file_name} is existed - Continue")
        continue 

    cap = cv2.VideoCapture(video_file)

    # Labeling
    frame_idx = 0
    labels = []
    landmarks_data = []
    
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5) as holistic:
        while cap.isOpened():            
            ret, frame = cap.read()
            
            if not ret:
                print("End of video or cannot read the video stream.")
                break

            # cv2.imshow('Video Frame', frame)
            
            
            frame_idx += 1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = holistic.process(image)

            frame_data = np.zeros((75, 3))  # 75 keypoints (33 pose, 21 right-hand, 21 left-hand), with 3 coordinates each (x, y, z)
            
            # Collect key points from the pose, hands (left and right)
            if results.pose_landmarks:
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    frame_data[idx] = [lm.x, lm.y, lm.z]

            # Check for right hand landmarks and append only if they exist
            if results.right_hand_landmarks:
                for idx, lm in enumerate(results.right_hand_landmarks.landmark):
                    frame_data[33 + idx] = [lm.x, lm.y, lm.z]  # Right hand starts after 33 pose landmarks
            else:
                frame_data[33:33 + 21] = 0  # Mark absent hand landmarks with 0


            # Check for left hand landmarks and append only if they exist
            if results.left_hand_landmarks:
                for idx, lm in enumerate(results.left_hand_landmarks.landmark):
                    frame_data[33 + 21 + idx] = [lm.x, lm.y, lm.z]  # Left hand starts after 33 pose + 21 right hand landmarks
            else:
                frame_data[33 + 21:33 + 42] = 0  # Mark absent hand landmarks with 0

            landmarks_data.append(frame_data)


    # Convert data to numpy arrays
    landmarks_array = np.array(landmarks_data)
    # labels_array = np.array(labels)

    # print(f"Shape landmarks: {landmarks_array.shape}")
    # print(f"Shape labels: {labels_array.shape}")

    # Save to npy file
    np.save(destiny_folder + f'/landmarks_{file_name}.npy', landmarks_array)
    # np.save(f'./temp/labels_{file_name}.npy', labels_array)
    # print("Error:",error)
    logger.info(f"Data in {file_name} saved to landmarks_{file_name}.npy")

    # Release resources
    cap.release()
    
cv2.destroyAllWindows()
