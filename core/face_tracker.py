import time
import math
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import config

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False


class StudentTrack:
    """Tracks an individual student's presence and visual behavior over time."""

    def __init__(self, track_id: str, name: Optional[str] = None, bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        self.track_id = track_id
        self.name = name or f"Student #{track_id.split('-')[-1]}"
        self.bbox = bbox  # x, y, w, h
        self.centroid = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        
        # Temporal Presence Metrics
        now = time.time()
        self.first_seen = now
        self.last_seen = now
        self.total_frames_seen = 1
        self.missed_frames = 0
        self.detection_timestamps: List[float] = [now]
        
        # Sustained Presence Tracking
        self.sustained_seconds = 0.0
        self.active_streak_seconds = 0.0
        self.last_streak_update = now
        self.attendance_status = "Pending"  # "Pending", "Present", "Partial", "Absent"
        self.sustained_presence_ratio = 0.0
        
        # Engagement & Behavioral Metrics
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.eye_openness = 1.0
        self.motion_energy = 0.0
        self.prev_centroid = self.centroid
        
        # Aggregated Engagement History
        self.current_engagement_score = 75.0
        self.engagement_history: List[float] = [75.0]
        self.is_attentive = True
        self.drowsy_count = 0
        self.distraction_count = 0
        self.inactivity_seconds = 0.0

    def update_position(self, bbox: Tuple[int, int, int, int], current_time: float):
        """Update tracker with new frame coordinates and adjust sustained presence."""
        self.bbox = bbox
        new_centroid = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        
        # Calculate spatial movement velocity
        dx = new_centroid[0] - self.prev_centroid[0]
        dy = new_centroid[1] - self.prev_centroid[1]
        self.motion_energy = math.hypot(dx, dy)
        self.prev_centroid = self.centroid
        self.centroid = new_centroid
        
        dt = current_time - self.last_seen
        self.last_seen = current_time
        self.total_frames_seen += 1
        self.missed_frames = 0
        self.detection_timestamps.append(current_time)
        
        # Sustained presence accumulator: require detections without huge gaps
        if dt <= 3.0:  # Continuous detection within 3-second tolerance
            self.active_streak_seconds += dt
            self.sustained_seconds += dt
        else:
            self.active_streak_seconds = 0.0  # Streak broken
            
        # Motion inactivity tracking
        if self.motion_energy < 1.5:
            self.inactivity_seconds += dt
        else:
            self.inactivity_seconds = 0.0

    def evaluate_attendance(self, session_elapsed_sec: float) -> str:
        """
        Evaluate sustained presence logic.
        Requires presence over sustained windows, avoiding false positives from passersby.
        """
        if session_elapsed_sec < 5.0:
            self.attendance_status = "Observing"
            self.sustained_presence_ratio = 1.0
            return self.attendance_status

        # Calculate presence ratio against total session elapsed time
        # Cap sustained seconds to elapsed seconds
        effective_sustained = min(self.sustained_seconds, session_elapsed_sec)
        self.sustained_presence_ratio = max(0.0, min(1.0, effective_sustained / session_elapsed_sec))

        # Check sustained presence thresholds
        # Must have at least MIN_DETECTION_STREAK_SEC to prevent passerby false positive
        if self.sustained_seconds < config.MIN_DETECTION_STREAK_SEC:
            self.attendance_status = "Transient (Passerby)"
        elif self.sustained_presence_ratio >= config.MIN_ATTENDANCE_RATIO_PRESENT:
            self.attendance_status = "Present"
        elif self.sustained_presence_ratio >= config.MIN_ATTENDANCE_RATIO_PARTIAL:
            self.attendance_status = "Partial Attendance"
        else:
            self.attendance_status = "Absent (< 30% presence)"

        return self.attendance_status

    def to_dict(self) -> dict:
        """Serialize student track for API and dashboard."""
        return {
            "track_id": self.track_id,
            "name": self.name,
            "bbox": list(self.bbox),
            "attendance_status": self.attendance_status,
            "sustained_presence_ratio": round(self.sustained_presence_ratio * 100, 1),
            "sustained_seconds": round(self.sustained_seconds, 1),
            "first_seen": time.strftime("%H:%M:%S", time.localtime(self.first_seen)),
            "last_seen": time.strftime("%H:%M:%S", time.localtime(self.last_seen)),
            "engagement_score": round(self.current_engagement_score, 1),
            "is_attentive": self.is_attentive,
            "pitch": round(self.pitch, 1),
            "yaw": round(self.yaw, 1),
            "roll": round(self.roll, 1),
            "drowsy_count": self.drowsy_count,
            "distraction_count": self.distraction_count,
            "inactivity_seconds": round(self.inactivity_seconds, 1)
        }


class FaceTracker:
    """
    Multimodal Face Detector & Multi-Person Spatial Tracker.
    Uses MediaPipe FaceLandmarker when available; falls back to OpenCV Haar Cascades.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.tracks: Dict[str, StudentTrack] = {}
        self.next_id_counter = 1
        self.session_start_time = time.time()
        self.roster: Dict[str, dict] = {}  # Enrolled student profiles
        
        # Initialize Detectors
        self.use_mediapipe = False
        self.landmarker = None
        
        default_model = Path(config.BASE_DIR) / "models" / "face_landmarker.task"
        active_model_path = model_path or str(default_model)
        
        if MEDIAPIPE_AVAILABLE and Path(active_model_path).exists():
            try:
                base_options = python.BaseOptions(model_asset_path=active_model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    num_faces=12
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                self.use_mediapipe = True
            except Exception as e:
                print(f"[FaceTracker] MediaPipe init warning: {e}. Falling back to Haar Cascade.")
                self.use_mediapipe = False

        # Haar Cascades Fallback
        haar_face_path = Path(config.BASE_DIR) / "models" / "haar" / "haarcascade_frontalface_default.xml"
        haar_eye_path = Path(config.BASE_DIR) / "models" / "haar" / "haarcascade_eye.xml"
        
        self.face_cascade = cv2.CascadeClassifier(str(haar_face_path))
        self.eye_cascade = cv2.CascadeClassifier(str(haar_eye_path))

    def reset_session(self):
        """Clear active tracks and restart session clock."""
        self.tracks.clear()
        self.next_id_counter = 1
        self.session_start_time = time.time()

    def set_roster(self, roster_dict: Dict[str, dict]):
        """Set known enrolled student profiles."""
        self.roster = roster_dict

    def detect_and_track(self, frame_bgr: np.ndarray, current_time: Optional[float] = None) -> List[StudentTrack]:
        """
        Process a video frame, detect all student faces, compute landmarks/pose,
        and update continuous tracking.
        """
        if current_time is None:
            current_time = time.time()
            
        h, w = frame_bgr.shape[:2]
        detected_candidates = []

        if self.use_mediapipe and self.landmarker is not None:
            detected_candidates = self._detect_mediapipe(frame_bgr, w, h)
        
        # Fallback to Haar Cascade if MediaPipe failed or returned 0 candidates
        if len(detected_candidates) == 0:
            detected_candidates = self._detect_haar(frame_bgr)

        # Multi-Object Association using IoU & Centroid Distance
        active_tracks = self._associate_detections_to_tracks(detected_candidates, current_time)

        # Update attendance evaluation for all tracks
        session_elapsed = current_time - self.session_start_time
        for track in self.tracks.values():
            track.evaluate_attendance(session_elapsed)

        return active_tracks

    def _detect_mediapipe(self, frame_bgr: np.ndarray, w: int, h: int) -> List[dict]:
        """Detect faces and extract 3D landmarks & transformation matrix via MediaPipe."""
        candidates = []
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.landmarker.detect(mp_image)

            if not result.face_landmarks:
                return candidates

            for i, face_lms in enumerate(result.face_landmarks):
                # Calculate bounding box from landmarks
                xs = [lm.x * w for lm in face_lms]
                ys = [lm.y * h for lm in face_lms]
                min_x, max_x = max(0, int(min(xs))), min(w, int(max(xs)))
                min_y, max_y = max(0, int(min(ys))), min(h, int(max(ys)))
                bw, bh = max_x - min_x, max_y - min_y
                
                if bw < 25 or bh < 25:
                    continue  # Filter out tiny noise artifacts

                # Extract Head Pose angles (Pitch, Yaw, Roll)
                pitch, yaw, roll = 0.0, 0.0, 0.0
                if result.facial_transformation_matrixes and i < len(result.facial_transformation_matrixes):
                    matrix = result.facial_transformation_matrixes[i]
                    pitch, yaw, roll = self._decompose_rotation_matrix(matrix)
                else:
                    # Heuristic pose approximation from landmark keypoints
                    pitch, yaw, roll = self._estimate_pose_from_landmarks(face_lms, w, h)

                # Eye blendshapes for blink / drowsiness
                eye_openness = 1.0
                if result.face_blendshapes and i < len(result.face_blendshapes):
                    blendshapes = {b.category_name: b.score for b in result.face_blendshapes[i]}
                    blink_left = blendshapes.get("eyeBlinkLeft", 0.0)
                    blink_right = blendshapes.get("eyeBlinkRight", 0.0)
                    eye_openness = max(0.0, 1.0 - (blink_left + blink_right) / 2.0)

                candidates.append({
                    "bbox": (min_x, min_y, bw, bh),
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll,
                    "eye_openness": eye_openness,
                    "landmarks": face_lms
                })
        except Exception as e:
            # MediaPipe runtime error fallback
            pass

        return candidates

    def _detect_haar(self, frame_bgr: np.ndarray) -> List[dict]:
        """OpenCV Haar cascade face detection fallback."""
        candidates = []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
            eye_openness = 1.0 if len(eyes) >= 1 else 0.4

            # Simple pose heuristic: horizontal eye symmetry
            yaw = 0.0
            if len(eyes) == 2:
                eye1_x = eyes[0][0] + eyes[0][2] // 2
                eye2_x = eyes[1][0] + eyes[1][2] // 2
                mid_eyes = (eye1_x + eye2_x) / 2.0
                yaw = ((mid_eyes - (w / 2.0)) / (w / 2.0)) * 30.0

            candidates.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "pitch": 0.0,
                "yaw": float(yaw),
                "roll": 0.0,
                "eye_openness": float(eye_openness),
                "landmarks": None
            })
        return candidates

    def _decompose_rotation_matrix(self, matrix: np.ndarray) -> Tuple[float, float, float]:
        """Decompose 4x4 or 3x3 transformation matrix into Pitch, Yaw, Roll in degrees."""
        try:
            r = np.array(matrix)
            if r.shape == (4, 4):
                r = r[:3, :3]
            sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
            singular = sy < 1e-6
            if not singular:
                pitch = math.atan2(r[2, 1], r[2, 2])
                yaw = math.atan2(-r[2, 0], sy)
                roll = math.atan2(r[1, 0], r[0, 0])
            else:
                pitch = math.atan2(-r[1, 2], r[1, 1])
                yaw = math.atan2(-r[2, 0], sy)
                roll = 0.0
            return (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))
        except Exception:
            return (0.0, 0.0, 0.0)

    def _estimate_pose_from_landmarks(self, landmarks, w: int, h: int) -> Tuple[float, float, float]:
        """Estimate yaw and pitch using key landmark geometry (Nose, Left Eye, Right Eye, Chin)."""
        try:
            nose = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            chin = landmarks[152]

            eye_center_x = (left_eye.x + right_eye.x) / 2.0
            eye_dist = abs(right_eye.x - left_eye.x)
            if eye_dist > 0.01:
                yaw = ((nose.x - eye_center_x) / eye_dist) * 60.0
            else:
                yaw = 0.0

            eye_center_y = (left_eye.y + right_eye.y) / 2.0
            face_height = abs(chin.y - eye_center_y)
            if face_height > 0.01:
                pitch = (((nose.y - eye_center_y) / face_height) - 0.45) * 80.0
            else:
                pitch = 0.0

            roll = math.degrees(math.atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x))
            return (float(pitch), float(yaw), float(roll))
        except Exception:
            return (0.0, 0.0, 0.0)

    def _associate_detections_to_tracks(self, candidates: List[dict], current_time: float) -> List[StudentTrack]:
        """Associate detections to existing tracks using IoU & spatial proximity."""
        matched_candidate_indices = set()
        active_tracks = []

        # Compare existing tracks with candidates
        for track_id, track in list(self.tracks.items()):
            best_iou = 0.0
            best_idx = -1

            for idx, cand in enumerate(candidates):
                if idx in matched_candidate_indices:
                    continue
                iou = self._calculate_iou(track.bbox, cand["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            # If strong match or close centroid distance
            if best_idx >= 0 and best_iou >= config.IOU_TRACKER_THRESHOLD:
                cand = candidates[best_idx]
                matched_candidate_indices.add(best_idx)
                track.update_position(cand["bbox"], current_time)
                track.pitch = cand["pitch"]
                track.yaw = cand["yaw"]
                track.roll = cand["roll"]
                track.eye_openness = cand["eye_openness"]
                active_tracks.append(track)
            else:
                track.missed_frames += 1
                # Check if track has been inactive too long
                if track.missed_frames > config.MAX_TRACK_AGE_FRAMES:
                    # Keep record in session history, but do not return as active this frame
                    pass
                else:
                    active_tracks.append(track)

        # Create new tracks for unmatched candidates
        for idx, cand in enumerate(candidates):
            if idx not in matched_candidate_indices:
                track_id = f"STU-{self.next_id_counter:03d}"
                self.next_id_counter += 1
                
                # Assign name if enrolled in roster
                assigned_name = self._match_or_assign_roster(track_id)
                new_track = StudentTrack(track_id, name=assigned_name, bbox=cand["bbox"])
                new_track.pitch = cand["pitch"]
                new_track.yaw = cand["yaw"]
                new_track.roll = cand["roll"]
                new_track.eye_openness = cand["eye_openness"]
                
                self.tracks[track_id] = new_track
                active_tracks.append(new_track)

        return active_tracks

    def _match_or_assign_roster(self, track_id: str) -> str:
        """Assign registered student name from roster if available."""
        if self.roster and track_id in self.roster:
            return self.roster[track_id].get("name", f"Student {track_id}")
        return f"Student {track_id.split('-')[-1]}"

    @staticmethod
    def _calculate_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes (x, y, w, h)."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        unionArea = float(boxAArea + boxBArea - interArea)

        if unionArea <= 0:
            return 0.0
        return interArea / unionArea
