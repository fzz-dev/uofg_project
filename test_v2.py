import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize variables
trajectory = []
shape = 'Unknown'

def calculate_angle(point1, point2, point3):
    angle = np.degrees(
        np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) -
        np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    )
    angle = np.abs(angle)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def analyze_trajectory(trajectory):
    if len(trajectory) < 3:
        return "Undetermined"

    # Fit polygon to the trajectory points
    peri = cv2.arcLength(np.array(trajectory), True)
    approx = cv2.approxPolyDP(np.array(trajectory), 0.04 * peri, True)

    num_vertices = len(approx)

    if num_vertices == 3:
        return "Triangle"
    elif num_vertices == 4:
        # Check if it's a square or rectangle
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        if 0.95 <= ar <= 1.05:
            return "Square"
        else:
            return "Rectangle"
    elif num_vertices > 4:
        return "Circle"
    else:
        return "Undetermined"


# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image and detect hands
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = image.shape
            index_finger_tip = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

            # Add to trajectory
            trajectory.append(index_finger_tip)
            for i in range(1, len(trajectory)):
                cv2.line(image, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

    # Analyze trajectory after collecting enough points
    if len(trajectory) > 50:  # Adjust the threshold as needed
        shape = analyze_trajectory(trajectory)
        trajectory = []  # Reset trajectory after analysis

    # Display the detected shape
    cv2.putText(image, f"Detected Shape: {shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                cv2.LINE_AA)
    # Display the resulting image
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
