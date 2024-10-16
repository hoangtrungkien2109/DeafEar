import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe holistic model
mp_holistic = mp.solutions.holistic

video_file = '/Users/trHien/PycharmProjects/ScrapeASLData/data/D0013.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_file)

landmarks_data = []

with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.1) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video or cannot read the video stream.")
            break

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
            # Ignore the right hand if not present
            frame_data[33:33 + 21] = np.nan  # Optionally mark absent hand landmarks with NaN

        # Check for left hand landmarks and append only if they exist
        if results.left_hand_landmarks:
            for idx, lm in enumerate(results.left_hand_landmarks.landmark):
                frame_data[33 + 21 + idx] = [lm.x, lm.y, lm.z]  # Left hand starts after 33 pose + 21 right hand landmarks
        else:
            # Ignore the left hand if not present
            frame_data[33 + 21:33 + 42] = np.nan  # Optionally mark absent hand landmarks with NaN

        landmarks_data.append(frame_data)

landmarks_array = np.array(landmarks_data)
print(f"Shape: {landmarks_array.shape}")
# Save to a npy file
np.save('landmarks_data4.npy', landmarks_array)

print("Landmarks data saved to landmarks_data.npy")

# Release resources
cap.release()
