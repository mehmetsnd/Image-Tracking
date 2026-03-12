import cv2
import os
import face_recognition
import numpy as np
import datetime
from ultralytics import YOLO
from sort import Sort

# ---------------- CONFIGURATION ----------------
KNOWN_FACES_DIR = "../known_faces"
ATTENDANCE_FILE = "../attendance.csv"
CONFIDENCE_THRESHOLD = 0.5
PROCESSING_WIDTH = 320  # Aggressive downscale for max FPS
# -----------------------------------------------

def load_known_faces():
    """
    Loads images from the known_faces directory and encodes them.
    """
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Warning: {KNOWN_FACES_DIR} does not exist. Create it and add images.")
        return [], []

    print("Loading known faces...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                # Load image
                image = face_recognition.load_image_file(image_path)
                # Compute encoding
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    # Use filename without extension as name
                    name = os.path.splitext(filename)[0]
                    known_face_names.append(name)
                    print(f"Loaded: {name}")
                else:
                    print(f"Skipping {filename}: No face found.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
    return known_face_encodings, known_face_names

def log_attendance(track_id, name):
    """
    Logs attendance with ID, Name, Date, Time columns.
    Logged ONLY on Save/Update events.
    """
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    file_exists = os.path.isfile(ATTENDANCE_FILE)
    
    with open(ATTENDANCE_FILE, "a") as f:
        if not file_exists:
            f.write("Student ID;Name;Date;Time\n")
        f.write(f"{track_id};{name};{date_str};{time_str}\n")
        
    print(f"Attendance Logged: {name} (ID: {track_id})")

def draw_corner_rect(img, bbox, color, length=20, thickness=2):
    """
    Draws a fancy corner-only rectangle.
    """
    x1, y1, x2, y2 = bbox
    
    # Top Left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    
    # Top Right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    
    # Bottom Left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    
    # Bottom Right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

def draw_fancy_label(img, text, pos, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    """
    Draws a text label with a semi-transparent background.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (t_w, t_h), _ = cv2.getTextSize(text, font, scale, thickness)
    
    x, y = pos
    padding = 5
    
    # Background rectangle
    cv2.rectangle(img, (x, y - t_h - padding), (x + t_w + padding * 2, y + padding), bg_color, -1)
    
    # Text
    cv2.putText(img, text, (x + padding, y), font, scale, text_color, thickness)

def main():
    # 1. Setup
    model = YOLO("yolov8n.pt")
    tracker = Sort(max_age=100, min_hits=3, iou_threshold=0.3)
    known_encodings, known_names = load_known_faces()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not found.")
        return

    track_id_to_name = {}

    # Window Setup (Resizable)
    window_name = "Advanced Classroom Attendance"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720) # Start with a reasonable default size

    print("Running... Press 'q' to quit, 's' to save, 'd' to delete, 'u' to update.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Optimization: Resize for processing
        h, w = frame.shape[:2]
        scale = PROCESSING_WIDTH / float(w)
        processed_h = int(h * scale)
        processed_frame = cv2.resize(frame, (PROCESSING_WIDTH, processed_h))

        # 2. Detection (YOLO on small frame)
        results = model(processed_frame, stream=True, classes=[0], verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Scale coordinates BACK to original size
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
                
                detections.append([x1, y1, x2, y2, conf])
        
        if len(detections) == 0:
            detections = np.empty((0, 5))
        else:
            detections = np.array(detections)
        
        # 3. Tracking (SORT on original coordinates)
        track_results = tracker.update(detections)
        
        # Optimization flag
        # Run Face Rec only once per second (approx 30 frames) instead of every 5.
        should_run_face_rec = (tracker.frame_count % 30 == 0)

        for track in track_results:
            x1, y1, x2, y2, track_id = map(int, track)
            track_id = int(track_id)
            
            name = "Unknown"
            color = (0, 0, 255) # Red
            
            # 4. Identification
            if track_id in track_id_to_name:
                name = track_id_to_name[track_id]
                color = (0, 255, 0) # Green
            elif should_run_face_rec:
                # Face Rec on ORIGINAL frame (better quality)
                # Ensure coords are within bounds
                face_x1 = max(0, x1)
                face_y1 = max(0, y1)
                face_x2 = min(w, x2)
                face_y2 = min(h, y2)
                
                face_img = frame[face_y1:face_y2, face_x1:face_x2]
                
                if face_img.size > 0:
                    # FPS Optimization: Resize large crops (e.g. body crops) before face detection
                    # HOG detection on large images is very slow.
                    # We limit the width of the crop passed to face_recognition.
                    fr_scale = 1.0
                    if face_img.shape[1] > 320:
                        fr_scale = 320.0 / face_img.shape[1]
                        face_img_small = cv2.resize(face_img, (0, 0), fx=fr_scale, fy=fr_scale)
                    else:
                        face_img_small = face_img

                    rgb_face = cv2.cvtColor(face_img_small, cv2.COLOR_BGR2RGB)
                    # Use 'hog' for speed
                    face_locations = face_recognition.face_locations(rgb_face, model="hog")
                    
                    if face_locations:
                        face_encs = face_recognition.face_encodings(rgb_face, face_locations)
                        if face_encs:
                            encoding = face_encs[0]
                            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=CONFIDENCE_THRESHOLD)
                            face_distances = face_recognition.face_distance(known_encodings, encoding)
                            
                            if len(face_distances) > 0:
                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index]:
                                    name = known_names[best_match_index]
                                    track_id_to_name[track_id] = name
                                    color = (0, 255, 0)

            # --- AUTO LOGGING REMOVED HERE ---

            # 5. UI Visualization (Fancy)
            draw_corner_rect(frame, (x1, y1, x2, y2), color)
            
            # Display Name and ID with background
            label = f"{name} (ID: {track_id})"
            draw_fancy_label(frame, label, (x1, y1 - 10), bg_color=color)

        # UI for Features
        cv2.putText(frame, "FPS Opt: ON (Resized)", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        # Detailed Instructions
        y_off = 40
        for line in ["'s': Save Unknown", "'d': Delete Selected", "'u': Update Selected", "'q': Quit"]:
            cv2.putText(frame, line, (10, y_off), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            y_off += 15

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        # --- COMMAND HANDLING ---
        elif key in [ord('s'), ord('d'), ord('u')]:
            # Find largest face (assumed to be the subject of interest)
            target_track = None
            max_area = 0
            
            for track in track_results:
                x1, y1, x2, y2, track_id = map(int, track)
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    target_track = track
            
            if target_track is not None:
                x1, y1, x2, y2, track_id = map(int, target_track)
                track_id = int(track_id)
                current_name = track_id_to_name.get(track_id)
                
                # DELETE Logic
                if key == ord('d'):
                    if current_name:
                        # Find file and delete
                        deleted = False
                        for f in os.listdir(KNOWN_FACES_DIR):
                            if os.path.splitext(f)[0] == current_name:
                                try:
                                    os.remove(os.path.join(KNOWN_FACES_DIR, f))
                                    print(f"DELETED: {current_name}")
                                    deleted = True
                                    break
                                except OSError as e:
                                    print(f"Error deleting file: {e}")
                        
                        if deleted:
                            # Force reload
                            known_encodings, known_names = load_known_faces()
                            # Remove from runtime cache
                            if track_id in track_id_to_name:
                                del track_id_to_name[track_id]
                    else:
                        print("Cannot delete: Entity is Unknown.")

                # SAVE (New) Logic
                elif key == ord('s'):
                    if current_name:
                         print(f"Already known as {current_name}. Use 'u' to update.")
                    else:
                        # Capture new
                        h, w, _ = frame.shape
                        face_img = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                        if face_img.size > 0:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"Person_{timestamp}.jpg"
                            filepath = os.path.join(KNOWN_FACES_DIR, filename)
                            cv2.imwrite(filepath, face_img)
                            print(f"SAVED: {filepath}")
                            
                            new_name = os.path.splitext(filename)[0]
                            log_attendance(track_id, new_name)

                            known_encodings, known_names = load_known_faces()

                # UPDATE (Existing) Logic
                elif key == ord('u'):
                    if not current_name:
                        print("Cannot update: Entity is Unknown. Use 's' to save first.")
                    else:
                        # Delete old
                        for f in os.listdir(KNOWN_FACES_DIR):
                            if os.path.splitext(f)[0] == current_name:
                                try:
                                    os.remove(os.path.join(KNOWN_FACES_DIR, f))
                                    print(f"Removed old entry for update: {current_name}")
                                    break
                                except OSError as e:
                                    print(f"Error removing old entry: {e}")
                        
                        # Save new
                        h, w, _ = frame.shape
                        face_img = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                        if face_img.size > 0:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"Person_{timestamp}.jpg"
                            filepath = os.path.join(KNOWN_FACES_DIR, filename)
                            cv2.imwrite(filepath, face_img)
                            print(f"UPDATED: {filepath}")
                            
                            new_name = os.path.splitext(filename)[0]
                            log_attendance(track_id, new_name)

                            known_encodings, known_names = load_known_faces()
                            if track_id in track_id_to_name:
                                 del track_id_to_name[track_id]

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
