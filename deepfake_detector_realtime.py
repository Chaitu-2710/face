import cv2
import numpy as np
from scipy.spatial import distance
import time
import sys
import os

class DeepfakeDetector:
    def __init__(self):
        # Load face detection model
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(f"Cascade file not found at {cascade_path}")
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            if not os.path.exists(eye_cascade_path):
                raise FileNotFoundError(f"Eye cascade file not found at {eye_cascade_path}")
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            
            if self.face_cascade.empty():
                raise Exception("Error loading face cascade classifier")
            if self.eye_cascade.empty():
                raise Exception("Error loading eye cascade classifier")
                
        except Exception as e:
            print(f"Error initializing cascade classifiers: {str(e)}")
            sys.exit(1)
            
        # Initialize video capture
        self.cap = None
        
        # Parameters for detection
        self.blink_threshold = 0.2
        self.movement_threshold = 10
        self.prev_face_center = None
        self.suspicious_count = 0
        self.frame_count = 0
        
    def calculate_eye_aspect_ratio(self, eye_points):
        """Calculate the eye aspect ratio to detect blinks"""
        try:
            if len(eye_points) < 6:
                return 1.0
                
            # Calculate vertical distances
            A = distance.euclidean(eye_points[1], eye_points[5])
            B = distance.euclidean(eye_points[2], eye_points[4])
            
            # Calculate horizontal distance
            C = distance.euclidean(eye_points[0], eye_points[3])
            
            # Calculate eye aspect ratio
            ear = (A + B) / (2.0 * C) if C > 0 else 1.0
            return ear
        except Exception as e:
            print(f"Error calculating eye aspect ratio: {str(e)}")
            return 1.0
        
    def detect_anomalies(self, frame):
        """Detect potential deepfake indicators in a frame"""
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame")
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # If no faces detected, show message
            if len(faces) == 0:
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return frame
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Calculate face center for movement detection
                face_center = np.array([x + w//2, y + h//2])
                
                # Detect eyes
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                # Check eye movement and blinking
                eye_detected = len(eyes) > 0
                if not eye_detected:
                    self.suspicious_count += 1
                    cv2.putText(frame, "No Eye Movement Detected!", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Check face movement
                if self.prev_face_center is not None:
                    movement = np.linalg.norm(face_center - self.prev_face_center)
                    if movement < self.movement_threshold:
                        self.suspicious_count += 1
                        cv2.putText(frame, "Low Face Movement!", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                self.prev_face_center = face_center
                
                # Draw confidence score
                confidence = max(0, min(100, 100 - (self.suspicious_count / max(1, self.frame_count)) * 100))
                cv2.putText(frame, f"Real Face Confidence: {confidence:.1f}%", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            return frame
            
        except Exception as e:
            print(f"Error in detect_anomalies: {str(e)}")
            return frame
    
    def start_detection(self, source=0):
        """Start real-time deepfake detection"""
        print("Initializing camera...")
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise Exception("Failed to open camera/video source")
            
            print("Camera initialized successfully!")
            print("Starting detection... Press 'q' to quit")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                self.frame_count += 1
                
                # Detect anomalies and draw results
                processed_frame = self.detect_anomalies(frame)
                
                # Display the frame
                cv2.imshow('Deepfake Detection', processed_frame)
                
                # Reset suspicious count periodically
                if self.frame_count % 30 == 0:
                    self.suspicious_count = max(0, self.suspicious_count - 1)
                
                # Break loop with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            
        finally:
            print("Cleaning up...")
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Cleanup complete")

if __name__ == "__main__":
    try:
        # Create and start the detector
        detector = DeepfakeDetector()
        detector.start_detection()
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
