import cv2
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import queue
import os

class MotionAnalyzer:
    def __init__(self, history_size=10):
        self.history_size = history_size
        self.motion_history = []
        self.last_frame = None
        
    def analyze(self, frame, face_roi):
        if self.last_frame is None:
            self.last_frame = frame
            return 0.5, 0.0  # Default scores
            
        try:
            # Calculate frame difference with improved noise handling
            frame_diff = cv2.absdiff(frame, self.last_frame)
            frame_diff = cv2.GaussianBlur(frame_diff, (5, 5), 0)  # Reduce noise
            motion_score = np.mean(frame_diff) / 255.0
            
            # Update history
            self.motion_history.append(motion_score)
            if len(self.motion_history) > self.history_size:
                self.motion_history.pop(0)
            
            # Analyze motion pattern with adjusted thresholds
            if len(self.motion_history) > 2:
                motion_std = np.std(self.motion_history)
                motion_mean = np.mean(self.motion_history[-3:])  # Look at recent motion
                
                # Real person: Natural variation in motion (adjusted thresholds)
                if 0.005 < motion_std < 0.15 and 0.01 < motion_mean < 0.25:
                    natural_score = 0.8
                # Static image: Very little motion
                elif motion_std < 0.005 and motion_mean < 0.008:
                    natural_score = 0.2
                # AI/Deepfake: Irregular motion patterns
                else:
                    natural_score = 0.4
                
                glitch_score = motion_std if motion_std > 0.15 else 0.0
            else:
                natural_score = 0.5
                glitch_score = 0.0
            
            self.last_frame = frame.copy()
            return natural_score, glitch_score
            
        except Exception as e:
            print(f"Error in motion analysis: {str(e)}")
            return 0.5, 0.0

class GlitchDetector:
    def __init__(self):
        self.last_features = None
        self.feature_detector = cv2.SIFT_create()
        
    def detect(self, frame):
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect features
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            
            if self.last_features is not None and descriptors is not None:
                # Match features
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(self.last_features, descriptors, k=2)
                
                # Calculate ratio of good matches
                good_matches = 0
                total_matches = 0
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches += 1
                    total_matches += 1
                
                if total_matches > 0:
                    match_ratio = good_matches / total_matches
                    
                    # High ratio indicates consistent features (real)
                    # Low ratio indicates inconsistent features (potential AI)
                    glitch_score = 1.0 - match_ratio
                else:
                    glitch_score = 0.5
            else:
                glitch_score = 0.0
            
            if descriptors is not None:
                self.last_features = descriptors
            
            return glitch_score
            
        except Exception as e:
            print(f"Error in glitch detection: {str(e)}")
            return 0.0

class DeepfakeDetector:
    def __init__(self):
        try:
            # Initialize detectors
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            if self.face_cascade.empty():
                raise Exception("Error loading face cascade classifier")
            
            # Initialize analyzers
            self.motion_analyzer = MotionAnalyzer()
            self.glitch_detector = GlitchDetector()
            
            # Detection history
            self.detection_history = []
            self.history_size = 30
            
        except Exception as e:
            print(f"Error initializing detector: {str(e)}")
            raise

    def analyze_face(self, frame, face):
        """Advanced face analysis with motion and glitch detection"""
        try:
            x, y, w, h = face
            face_roi = frame[y:y+h, x:x+w]
            
            # Ensure minimum face size for reliable detection
            if w < 100 or h < 100:
                return 0.3, "Too Far"
                
            # Ensure face isn't too close
            if w > frame.shape[1] * 0.8 or h > frame.shape[0] * 0.8:
                return 0.3, "Too Close"
            
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # 1. Motion Analysis with distance-based adjustment
            motion_score, motion_glitch = self.motion_analyzer.analyze(frame, face_roi)
            
            # Adjust motion sensitivity based on face size
            size_factor = min(1.0, max(0.3, (w * h) / (200 * 200)))
            motion_score = motion_score * size_factor
            
            # 2. Eye Detection with improved parameters
            eyes = self.eye_cascade.detectMultiScale(
                gray_roi,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(int(w/15), int(h/15)),
                maxSize=(int(w/3), int(h/3))
            )
            
            eye_score = 1.0 if len(eyes) >= 2 else 0.3
            
            # 3. Texture Analysis with distance compensation
            laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
            texture_var = np.var(laplacian)
            texture_score = min(1.0, (texture_var / 1000) * (1 + (1 - size_factor)))
            
            # 4. Skin Texture Analysis
            ycrcb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
            skin_mask = cv2.inRange(ycrcb_roi[:,:,1], 133, 173) & cv2.inRange(ycrcb_roi[:,:,2], 77, 127)
            skin_ratio = np.sum(skin_mask) / (skin_mask.shape[0] * skin_mask.shape[1])
            skin_score = min(1.0, skin_ratio * 1.5)
            
            # Calculate weighted score with distance compensation
            realness_score = (
                0.35 * motion_score +      # Motion naturalness
                0.25 * eye_score +         # Eye presence
                0.20 * texture_score +     # Texture naturalness
                0.20 * skin_score         # Skin texture
            )
            
            # Adjust thresholds based on face size
            if size_factor < 0.5:  # Face is far
                confidence_threshold = 0.5
            else:
                confidence_threshold = 0.6
            
            # Improved classification
            if motion_glitch > 0.4:
                detection_type = "AI Generated"
                confidence = 0.3
            elif motion_score < 0.2 and texture_var < 100:
                detection_type = "Static Image"
                confidence = 0.3
            elif realness_score > confidence_threshold and eye_score > 0.5:
                detection_type = "Real Person"
                confidence = 0.67
            else:
                detection_type = "Uncertain"
                confidence = realness_score
            
            # Update detection history
            self.detection_history.append((confidence, detection_type))
            if len(self.detection_history) > self.history_size:
                self.detection_history.pop(0)
            
            # Use majority voting with recency bias
            if len(self.detection_history) > 5:
                recent_types = [t[1] for t in self.detection_history[-5:]]
                recent_conf = [c[0] for c in self.detection_history[-5:]]
                
                # Weight recent detections more heavily
                type_counts = {}
                for i, t in enumerate(recent_types):
                    weight = (i + 1) / len(recent_types)  # More recent = higher weight
                    type_counts[t] = type_counts.get(t, 0) + weight
                
                final_type = max(type_counts.items(), key=lambda x: x[1])[0]
                avg_conf = np.average(recent_conf, weights=range(1, len(recent_conf) + 1))
                
                return avg_conf, final_type
            
            return confidence, detection_type
            
        except Exception as e:
            print(f"Error in face analysis: {str(e)}")
            return 0.3, "Error"

    def detect_deepfake(self, frame):
        """Main detection method"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Improved face detection parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(100, 100),
                maxSize=(int(frame.shape[1] * 0.8), int(frame.shape[0] * 0.8))
            )
            
            if len(faces) > 0:
                # Sort faces by size and pick the largest
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                face = faces[0]
                confidence, status = self.analyze_face(frame, face)
                return confidence, face, status
            
            return None, None, "No Face Detected"
            
        except Exception as e:
            print(f"Error in detection: {str(e)}")
            return None, None, "Error"

class DeepfakeDetectorGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Advanced Deepfake Detector")
        
        try:
            self.detector = DeepfakeDetector()
            self.create_widgets()
            
            self.cap = None
            self.is_running = False
            self.frame_queue = queue.Queue(maxsize=1)
            
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))
            self.window.destroy()
    
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video display
        self.video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="5")
        self.video_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0)
        
        # Controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        controls_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Start/Stop button
        self.start_button = ttk.Button(
            controls_frame,
            text="Start Detection",
            command=self.toggle_detection,
            width=20
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Status display
        status_frame = ttk.LabelFrame(controls_frame, text="Detection Status", padding="5")
        status_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Detection type
        self.type_var = tk.StringVar(value="Type: Not Running")
        type_label = ttk.Label(status_frame, textvariable=self.type_var, font=('Arial', 10, 'bold'))
        type_label.grid(row=0, column=0, padx=5, sticky=tk.W)
        
        # Confidence display
        confidence_frame = ttk.Frame(status_frame)
        confidence_frame.grid(row=1, column=0, padx=5, sticky=(tk.W, tk.E))
        
        ttk.Label(confidence_frame, text="Confidence:").grid(row=0, column=0, padx=(0,5))
        self.confidence_var = tk.DoubleVar(value=0.0)
        self.confidence_bar = ttk.Progressbar(
            confidence_frame,
            length=200,
            mode='determinate',
            variable=self.confidence_var
        )
        self.confidence_bar.grid(row=0, column=1)
        
        # Create style for colored labels
        self.style = ttk.Style()
        self.style.configure("Real.TLabel", foreground="green")
        self.style.configure("Fake.TLabel", foreground="red")
        self.style.configure("Unknown.TLabel", foreground="orange")
        
        # Detection details
        details_frame = ttk.LabelFrame(main_frame, text="Detection Details", padding="5")
        details_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.details_var = tk.StringVar(value="Waiting for detection...")
        self.details_label = ttk.Label(details_frame, textvariable=self.details_var, wraplength=400)
        self.details_label.grid(row=0, column=0, padx=5, pady=5)
    
    def update_detection_display(self, detection_type, confidence):
        # Update confidence
        self.confidence_var.set(confidence * 100)
        
        # Update type with color
        if detection_type == "Real Person":
            style = "Real.TLabel"
            details = " Natural motion detected\n Consistent features\n Natural texture"
        elif detection_type == "AI Generated":
            style = "Fake.TLabel"
            details = " Irregular motion patterns\n Visual glitches detected\n Inconsistent features"
        elif detection_type == "Static Image":
            style = "Fake.TLabel"
            details = " No motion detected\n Static features\n Possible photo"
        else:
            style = "Unknown.TLabel"
            details = " Uncertain detection\n Mixed signals\n Need more data"
        
        self.type_var.set(f"Type: {detection_type}")
        self.details_var.set(details)
        self.details_label.configure(style=style)
    
    def toggle_detection(self):
        if not self.is_running:
            self.start_detection()
        else:
            self.stop_detection()
    
    def start_detection(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")
            
            self.is_running = True
            self.start_button.config(text="Stop Detection")
            self.type_var.set("Type: Running...")
            
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            self.update_gui()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.stop_detection()
    
    def stop_detection(self):
        self.is_running = False
        self.start_button.config(text="Start Detection")
        self.type_var.set("Type: Stopped")
        self.details_var.set("Waiting for detection...")
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def detection_loop(self):
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception("Failed to grab frame")
                
                result = self.detector.detect_deepfake(frame)
                if result[0] is not None:
                    confidence, face, detection_type = result
                    x, y, w, h = face
                    
                    # Draw rectangle
                    color = (0, 255, 0) if detection_type == "Real Person" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Update display
                    self.update_detection_display(detection_type, confidence)
                
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    continue
                    
            except Exception as e:
                print(f"Error in detection loop: {str(e)}")
                self.stop_detection()
                break
    
    def update_gui(self):
        if self.is_running:
            try:
                frame = self.frame_queue.get_nowait()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                
                self.video_label.img_tk = img_tk
                self.video_label.configure(image=img_tk)
                
            except queue.Empty:
                pass
            
            self.window.after(10, self.update_gui)

def main():
    try:
        root = tk.Tk()
        app = DeepfakeDetectorGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
