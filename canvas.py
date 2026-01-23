import cv2
import numpy as np
import mediapipe as mp
import traceback

try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    colorIndex = 0
    brush_thickness = 15

    # Initialize canvas with a default size, will be resized if needed
    canvas = np.full((720, 1280, 3), 255, dtype=np.uint8)
    
    cap = cv2.VideoCapture(1)
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
        
        # Draw UI
        cv2.rectangle(frame, (40, 1), (140, 80), (0, 0, 0), 2)
        cv2.rectangle(frame, (160, 1), (260, 80), (0, 0, 0), 2)
        cv2.rectangle(frame, (280, 1), (380, 80), (0, 0, 0), 2)
        cv2.rectangle(frame, (400, 1), (500, 80), (0, 0, 0), 2)
        
        cv2.rectangle(frame, (40, 1), (140, 80), colors[0], -1)
        cv2.rectangle(frame, (160, 1), (260, 80), colors[1], -1)
        cv2.rectangle(frame, (280, 1), (380, 80), colors[2], -1)
        cv2.rectangle(frame, (400, 1), (500, 80), colors[3], -1)

        # Size buttons (Small, Medium, Large)
        cv2.circle(frame, (800, 40), 40, (0, 0, 0), 2) 
        cv2.circle(frame, (800, 40), 10, (100, 100, 100), -1) # Small 10px

        cv2.circle(frame, (900, 40), 40, (0, 0, 0), 2)
        cv2.circle(frame, (900, 40), 20, (100, 100, 100), -1) # Medium 20px

        cv2.circle(frame, (1000, 40), 40, (0, 0, 0), 2)
        cv2.circle(frame, (1000, 40), 30, (100, 100, 100), -1) # Large 30px
        
        cv2.putText(frame, "SIZES", (880, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

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
                
                # Check fingers up (Index, Middle, Ring, Pinky)
                fingers = []
                fingers.append(1 if lm8.y < hand_landmarks.landmark[6].y else 0)
                fingers.append(1 if lm12.y < hand_landmarks.landmark[10].y else 0)
                fingers.append(1 if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y else 0)
                fingers.append(1 if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y else 0)

                if all(fingers): # Palm (4 fingers up) -> Clear
                    canvas[:] = 255
                    cv2.putText(frame, "CLEARED", (cx, cy-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                elif fingers[0] and fingers[1]: # Selection Mode
                    prev_x, prev_y = 0, 0
                    if cy < 90: # Check selection
                        if 40 < cx < 140: colorIndex = 0
                        elif 160 < cx < 260: colorIndex = 1
                        elif 280 < cx < 380: colorIndex = 2
                        elif 400 < cx < 500: colorIndex = 3
                        # Check Size Selection
                        elif 760 < cx < 840: brush_thickness = 10
                        elif 860 < cx < 940: brush_thickness = 20
                        elif 960 < cx < 1040: brush_thickness = 30

                    cv2.circle(frame, (cx, cy), 10, colors[colorIndex] if colorIndex >= 0 else (0,0,0), cv2.FILLED)

                elif fingers[0] and not fingers[1]: # Draw Mode
                    col = colors[colorIndex] if colorIndex >= 0 else (255, 255, 255)
                    
                    # --- Smoothing Logic ---
                    alpha = 0.5
                    if 'smooth_cx' not in locals(): smooth_cx, smooth_cy = cx, cy
                    if prev_x == 0 and prev_y == 0: smooth_cx, smooth_cy = cx, cy
                    smooth_cx = int(smooth_cx * (1 - alpha) + cx * alpha)
                    smooth_cy = int(smooth_cy * (1 - alpha) + cy * alpha)
                    curr_x, curr_y = smooth_cx, smooth_cy

                    # Visual feedback for cursor
                    cv2.circle(frame, (curr_x, curr_y), brush_thickness, col if colorIndex >= 0 else (0,0,0), cv2.FILLED)
                    
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = curr_x, curr_y
                    
                    cv2.line(canvas, (prev_x, prev_y), (curr_x, curr_y), col, brush_thickness)
                    
                    prev_x, prev_y = curr_x, curr_y
                else:
                    prev_x, prev_y = 0, 0

        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv_mask = cv2.threshold(canvas_gray, 250, 255, cv2.THRESH_BINARY_INV)
        inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, cv2.bitwise_not(inv_mask))
        drawing = cv2.bitwise_and(canvas, inv_mask)
        frame = cv2.add(frame, drawing)        
        
        cv2.imshow("Air Canvas", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

except Exception:
    traceback.print_exc()