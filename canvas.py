import cv2
import numpy as np
import mediapipe as mp
import traceback

try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize canvas with a default size, will be resized if needed
    canvas = np.full((720, 1280, 3), 255, dtype=np.uint8)
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    prev_x, prev_y = 0, 0
    
    print("Canvas Initialised, Raise finger to draw")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Warning: Failed to read frame from camera. Exiting...")
            break

        # Flip the image horizontally for a later selfie-view display
        frame = cv2.flip(image, 1)
        h, w, c = frame.shape

        # Ensure canvas matches frame size
        if canvas.shape[:2] != (h, w):
            print(f"Adjusting canvas size from {canvas.shape[:2]} to {(h, w)}")
            canvas = cv2.resize(canvas, (w, h))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm8 = hand_landmarks.landmark[8]  # Index finger tip
                lm12 = hand_landmarks.landmark[12] # Middle finger tip

                cx, cy = int(lm8.x * w), int(lm8.y * h)
                
                # Check if fingers are up
                index_up = lm8.y < hand_landmarks.landmark[6].y
                middle_up = lm12.y < hand_landmarks.landmark[10].y

                # Drawing logic: If both Index and Middle fingers are up
                # Note: Adjust logic here if you want only Index finger to draw
                if index_up and middle_up:
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = cx, cy
                    
                    # Draw on canvas
                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 0, 0), 15)
                    prev_x, prev_y = cx, cy
                else:
                    prev_x, prev_y = 0, 0
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Combine frame and canvas
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, canvas_thresh = cv2.threshold(canvas_gray, 50, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.cvtColor(canvas_thresh, cv2.COLOR_GRAY2BGR)
        
        frame = cv2.bitwise_and(frame, inv_mask)
        frame = cv2.addWeighted(frame, 1, canvas, 0.5, 0)        
        
        cv2.imshow("Air Canvas", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

except Exception:
    traceback.print_exc()