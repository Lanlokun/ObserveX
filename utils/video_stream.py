import cv2
import mediapipe as mp
from models.load_model import load_yolo_model

model = load_yolo_model()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
draw_utils = mp.solutions.drawing_utils

def classify_activity(landmarks):
    if landmarks:
        if landmarks[23].y < landmarks[11].y:
            return "Standing"
        else:
            return "Sitting"
    return "Unknown"

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))

        results = model(frame)
        annotated_frame = results[0].plot() 

        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(rgb_frame)

        if results_pose.pose_landmarks:
            draw_utils.draw_landmarks(
                annotated_frame,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            activity = classify_activity(results_pose.pose_landmarks.landmark)
            cv2.putText(annotated_frame, f"Activity: {activity}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
