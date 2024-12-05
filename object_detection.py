import torch
import cv2
from gtts import gTTS
import os
import time
import threading
import platform

# Load the YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()  # Set to evaluation mode

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use '0' for the default camera

# Focal length for distance estimation (adjust based on your camera setup)
FOCAL_LENGTH = 615  # in pixels (calibrated value)

def estimate_distance(known_width, object_width_in_pixels):
    """
    Estimate the distance of an object from the camera.
    :param known_width: Real-world width of the object in cm (or inches)
    :param object_width_in_pixels: Width of the object in pixels
    :return: Distance in cm
    """
    if object_width_in_pixels > 0:
        return (known_width * FOCAL_LENGTH) / object_width_in_pixels
    return -1

def detect_objects(frame):
    # Resize and prepare the frame for YOLOv5
    img_resized = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Run object detection
    results = model(img_rgb)
    return results.pandas().xyxy[0]  # Returns detected objects in a DataFrame

def speak(text):
    """
    Converts text to speech and plays it.
    """
    tts = gTTS(text=text, lang='en')
    audio_file = "detected_object.mp3"
    tts.save(audio_file)
    if platform.system() == "Windows":
        os.system(f"start {audio_file}")
    elif platform.system() == "Darwin":  # macOS
        os.system(f"afplay {audio_file}")
    else:  # Linux
        os.system(f"mpg321 {audio_file}")
    time.sleep(2)  # Wait for audio playback to finish
    os.remove(audio_file)

def speak_async(text):
    threading.Thread(target=speak, args=(text,)).start()

def main_loop():
    last_object_spoken = ""  # Keep track of the last spoken object to avoid repetition

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Detect objects in the current frame
        detected_objects = detect_objects(frame)

        for _, obj in detected_objects.iterrows():
            obj_name = obj['name']
            confidence = obj['confidence']
            x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
            object_width = x2 - x1

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{obj_name} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Estimate and display distance
            distance = estimate_distance(known_width=15, object_width_in_pixels=object_width)  # Assume 15 cm width
            if distance > 0:
                cv2.putText(frame, f"Dist: {distance:.2f} cm", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # Speak the name and distance if it's new or changed significantly
                if confidence > 0.5 and (obj_name != last_object_spoken or abs(distance - 50) > 10):
                    speak_async(f"I see a {obj_name} approximately {distance:.2f} centimeters away.")
                    last_object_spoken = obj_name

        # Display the frame with annotations
        cv2.imshow("Object Detection with Distance", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Run the main detection and feedback loop
if __name__ == "__main__":
    main_loop()
