import cv2
import mediapipe as mp
import pyautogui
import random
from pynput.mouse import Button, Controller
import math

# Initialize mouse controller
mouse = Controller()

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Function to calculate distance between two landmarks
def get_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# Check if all fingers are closed
def all_fingers_closed(landmarks):
    # Finger tips (4, 8, 12, 16, 20) and their respective bases (2, 5, 9, 13, 17)
    finger_tip_base_pairs = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)]

    for tip, base in finger_tip_base_pairs:
        distance = get_distance(landmarks[tip], landmarks[base])
        if distance > 0.05:  # Threshold for 'closed' fingers
            return False
    return True

# Gesture detection
def detect_gesture(frame, landmarks):
    if len(landmarks) >= 21:
        # Extract necessary landmarks
        index_tip = landmarks[8]
        thumb_tip = landmarks[4]
        middle_tip = landmarks[12]
        little_tip = landmarks[20]

        # Convert to pixel positions
        screen_x = int(index_tip[0] * screen_width)
        screen_y = int(index_tip[1] * screen_height / 2)

        # Move mouse based on index finger position
        pyautogui.moveTo(screen_x, screen_y)

        # Calculate distances
        thumb_index_dist = get_distance(thumb_tip, index_tip)

        # Left Click (Thumb and Index finger close)
        if thumb_index_dist < 0.05:
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Right Click (Thumb close to Middle Finger)
        elif get_distance(thumb_tip, middle_tip) < 0.05:
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Scroll Up and Down (Middle Finger movement)
        if landmarks[8][1] < landmarks[6][1] - 0.05:  # Move Up
            pyautogui.scroll(10)
            cv2.putText(frame, "Scroll Up", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif landmarks[8][1] > landmarks[6][1] + 0.05:  # Move Down
            pyautogui.scroll(-10)
            cv2.putText(frame, "Scroll Down", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)

        # Take Screenshot (Thumb and Little Finger close)
        thumb_pinky_dist = get_distance(thumb_tip, little_tip)
        if thumb_pinky_dist < 0.05:  # Adjust threshold if necessary
            screenshot = pyautogui.screenshot()
            label = random.randint(1, 1000)
            screenshot.save(f"my_screenshot_{label}.png")
            cv2.putText(frame, "Screenshot Taken", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

# Main function
def main():
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip and convert frame to RGB
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            result = hands.process(rgb_frame)
            landmarks_list = []

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get landmark coordinates
                    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    landmarks_list = landmarks

                # Detect gestures
                detect_gesture(frame, landmarks_list)

            # Show the frame
            cv2.imshow("Virtual Mouse Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()
